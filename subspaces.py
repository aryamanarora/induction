# This file is mostly copied directly from https://github.com/mlab-account/mlab2/blob/subspace/subspace/expr_d3_induction.py
# Much of the functionality is duplicated by e.g. model.py, although it is unclear at the moment how interoperable the two
# implementations are (there are definitely some differences).
#
# Should cleanup soon.
#
# When refactoring, see if it's possible to more tightly couple the generation of the projection matrices with the experiments
# which use them. As of now, nothing prevents someone from writing a bad matcher on the experiment side and putting the matrix
# somewhere it doesn't belong.
# %%

# Experiments using the two-layer transformer on induction heads

from model import load_model_and_data

import multiprocessing as mp
mp.set_start_method('fork')

from supsub.utils import (
    get_default_device, get_data_dir, iter_dataset_infinity, init_torch_seed)

from supsub.solver import (
    LookAheadProjectionSolver, CircuitWithProjection)
from supsub.rc_ext import (expand_with_batch, ScrubPrinter, printer,
                           clear_cuda_cache)

import rust_circuit as rc

from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.tools.data_loading import get_val_seqs
from interp.circuit.interop_rust.module_library import (
    load_model_id, negative_log_likelyhood)
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import pickle
import itertools
from pathlib import Path
from typing import Optional, Union

# %%
# Load data

DEVICE = get_default_device()

LOGFILE= Path(__file__).with_suffix('.log')
with LOGFILE.open('w'):
    # create and truncate the log
    pass

def logprint(*args):
    """print to the log file"""
    with LOGFILE.open('a') as fout:
        print(*args, file=fout)
    print(*args)    # also print to stdout

seq_len = 300  # longer seq len is better, but short makes stuff a bit easier...

def load_toks(train: bool, start: int, num: int, name: str) -> rc.Array:
    tok_cache_file = get_data_dir() / f'{name}_cache.pth'
    if tok_cache_file.exists():
        print('load toks from file')
        dataset_toks = torch.load(tok_cache_file)
    else:
        dataset_toks = torch.tensor(
            get_val_seqs(train=train, n_files=num, files_start=start, max_size=seq_len + 1)
        )
        torch.save(dataset_toks, tok_cache_file)
    dataset_toks = dataset_toks.to(DEVICE)
    print(f'loaded dataset {name}: {dataset_toks.shape}')
    return rc.Array(dataset_toks, name)

toks_int_values = load_toks(True, 0, 100, 'toks_int_values')
toks_dataset = TensorDataset(toks_int_values.value.to('cpu'))
toks_int_values_validation = load_toks(False, 0, 25, 'toks_int_values_validation')

CACHE_DIR = f'{RRFS_DIR}/ryan/induction_scrub/cached_vals'
good_induction_candidate: torch.Tensor = (
    torch.load(f'{CACHE_DIR}/induction_candidates_2022-10-15 04:48:29.970735.pt')
    .to(device=DEVICE)
)

print('good ind rate',
      good_induction_candidate[toks_int_values.value].float().mean())

toks_int_values = rc.cast_circuit(
    toks_int_values,
    rc.TorchDeviceDtypeOp(device=DEVICE, dtype='int64')).cast_array()

# sampled input
toks_int_var = rc.DiscreteVar(toks_int_values, name='toks_int_var')

# inputs and expected outputs
input_toks = toks_int_var.index(I[:-1], name='input_toks_int')
true_toks = toks_int_var.index(I[1:], name='true_toks_int')
assert input_toks.shape == (seq_len,)  # type: ignore
assert true_toks.shape == (seq_len,)  # type: ignore

# %%

# Model configuration from day 5

def load_and_transform_circuit(model_id: str):
    """load a gpt circuit and do some housekeeping

    :return: circuit, tokenizer
    """
    circ_dict, tokenizer, model_info = load_model_id(model_id)
    circ_dict = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE))
                 for s, c in circ_dict.items()}
    print('first example:',
          tokenizer.batch_decode(toks_int_values.value[:1].int()))

    tok_embeds = circ_dict['t.w.tok_embeds']
    pos_embeds = circ_dict['t.w.pos_embeds']
    idxed_embeds = rc.GeneralFunction.gen_index(
        tok_embeds, input_toks, index_dim=0, name='idxed_embeds')

    assert model_info.causal_mask, (
        'Should not apply causal mask if the transformer does not expect it!')
    causal_mask = rc.Array(
        (torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]).to(
            tok_embeds.cast_array().value),
        f't.a.c.causal_mask',
    )
    assert model_info.pos_enc_type == 'shortformer'
    pos_embeds = pos_embeds.index(I[:seq_len], name='t.w.pos_embeds_idxed')
    model = rc.module_new_bind(
        circ_dict['t.bind_w'],
        ('t.input', idxed_embeds),
        ('a.mask', causal_mask),
        ('a.pos_input', pos_embeds),
        name='t.call',
    )
    assert not rc.get_free_symbols(model)

    transformed_circuit = model.update(
        't.bind_w',
        lambda c: configure_transformer(
            c,
            To.ATTN_HEAD_MLP_NORM,
            split_by_head_config='full',
            use_pull_up_head_split=True,
            use_flatten_res=True,
        ),
    )
    transformed_circuit = rc.conform_all_modules(transformed_circuit)
    subbed_circuit = transformed_circuit.cast_module().substitute()
    subbed_circuit = subbed_circuit.rename("logits")
    def module_but_norm(circuit: rc.Circuit) -> bool:
        """Match all Module nodes that are not layer norms"""
        return circuit.is_module() and not (
            "norm" in circuit.name or "ln" in circuit.name or "final" in circuit.name
        )

    while True:
        prev = subbed_circuit
        subbed_circuit = subbed_circuit.update(module_but_norm, lambda c: c.cast_module().substitute())
        if prev == subbed_circuit:
            break

    print('remaining modules:',
          sorted([i.name for i in subbed_circuit.get(rc.Module)]))

    renamed_circuit = subbed_circuit.update(
        rc.Regex(r'[am]\d(.h\d)?$'), lambda c: c.rename(c.name + '.inner'))
    renamed_circuit = renamed_circuit.update(
        't.inp_tok_pos', lambda c: c.rename('embeds'))

    for l in range(model_info.params.num_layers):
        # b0 -> a1.input, ... b11 -> final.input
        nxt = 'final' if l == model_info.params.num_layers - 1 else f'a{l+1}'
        renamed_circuit = renamed_circuit.update(
            f'b{l}', lambda c: c.rename(f'{nxt}.input'))

        # b0.m -> m0, etc.
        renamed_circuit = renamed_circuit.update(
            f'b{l}.m', lambda c: c.rename(f'm{l}'))
        renamed_circuit = renamed_circuit.update(
            f'b{l}.m.p_bias', lambda c: c.rename(f'm{l}.p_bias'))
        renamed_circuit = renamed_circuit.update(
            f'b{l}.a', lambda c: c.rename(f'a{l}'))
        renamed_circuit = renamed_circuit.update(
            f'b{l}.a.p_bias', lambda c: c.rename(f'a{l}.p_bias'))

        for h in itertools.count():
            # b0.a.h0 -> a0.h0, etc.
            src_name = f'b{l}.a.h{h}'
            if renamed_circuit.get(src_name):
                renamed_circuit = renamed_circuit.update(
                    src_name, lambda c: c.rename(f'a{l}.h{h}'))
            else:
                break

    loss = rc.Module(
        negative_log_likelyhood.spec,
        **{"ll.input": renamed_circuit, "ll.label": true_toks},
        name="t.loss",
    )
    is_good_induction_candidate = rc.GeneralFunction.gen_index(
        x=rc.Array(good_induction_candidate.to(torch.float32),
                   name='tok_is_induct_candidate'),
        index=input_toks,
        index_dim=0,
        name='induct_candidate',
    )
    loss = rc.Einsum(
        (loss, (0,)),
        (is_good_induction_candidate, (0,)),
        out_axes=(0,),
        name='loss_on_candidates',
    )
    expected_loss_by_seq = rc.Cumulant(loss, name='t.expected_loss_by_seq')
    expected_loss = expected_loss_by_seq.mean(
        name='t.expected_loss', scalar_name='recip_seq')

    expected_loss.print(printer)

    expected_loss = rc.cast_circuit(expected_loss,
                                    rc.TorchDeviceDtypeOp(device=DEVICE))
    return expected_loss, tokenizer

circuit, tokenizer = load_and_transform_circuit(model_id='attention_only_2')

# %%

toks_int_var_other = rc.DiscreteVar(toks_int_values, name='toks_int_var_other')

g_rng = np.random.default_rng(42)
class PreshuffledTensorDataset(Dataset):
    def __init__(self, val: torch.Tensor, size: int):
        self.val = val
        self.remap = np.arange(val.shape[0])
        g_rng.shuffle(self.remap)
        self.remap = self.remap[:size]

    def __len__(self):
        return len(self.remap)

    def __getitem__(self, i):
        return self.val[self.remap[i]]

toks_dataset_test = PreshuffledTensorDataset(
    toks_int_values_validation.value.to('cpu'), 256 * 20)
toks_dataset_test_other = PreshuffledTensorDataset(
    toks_int_values_validation.value.to('cpu'), 256 * 20)

@torch.no_grad()
def sample_and_evaluate(c: rc.Circuit, batch_size=256) -> float:
    """run a circuit of scalar shape on the sampler"""
    # we use a direct for loop instead of rc.Sampler for efficiency reasons
    has_other_inp = c.are_any_found(toks_int_var_other)

    inp_names = [toks_int_var.name]
    if has_other_inp:
        inp_names.append(toks_int_var_other.name)
        c, (sym_inp, sym_inp_other) = expand_with_batch(
            c, inp_names, batch_size)
    else:
        c, sym_inp = expand_with_batch(c, inp_names[0], batch_size)

    clear_cuda_cache()
    schedule = rc.optimize_to_schedule(c, settings=rc.OptimizationSettings(
        max_memory=32*2**30, max_single_tensor_memory=32*2**30
    ))

    loader = DataLoader(toks_dataset_test, batch_size=batch_size,
                        num_workers=2, drop_last=True)
    loader_other = DataLoader(toks_dataset_test_other, batch_size=batch_size,
                              num_workers=2, drop_last=True)
    tot_loss = 0
    tot_size = 0
    for inp, inp_other in zip(tqdm(loader, desc='test'), loader_other):
        inp = inp.to(DEVICE)
        inp_other = inp_other.to(DEVICE)
        repl= {sym_inp.hash: inp}
        if has_other_inp:
            repl[sym_inp_other.hash] = inp_other
        out = schedule.replace_tensors(repl).evaluate()
        tot_loss += out.sum().item()
        tot_size += out.shape[0]

    return tot_loss / tot_size

baseline_loss = sample_and_evaluate(circuit)
print(f'{baseline_loss=}')
torch.testing.assert_close(baseline_loss, 0.17, atol=1e-2, rtol=1e-3)

def scrub_input(c: rc.Circuit, in_path: rc.IterativeMatcher) -> rc.Circuit:
    c1 = c.update(in_path.chain('toks_int_var'), lambda _: toks_int_var_other)
    assert c1 != c
    return c1

all_rep_circuit = scrub_input(circuit, rc.IterativeMatcher('final.call'))
print('random baseline', sample_and_evaluate(all_rep_circuit))

# %%

ALL_NODES_NAMES = set(f'a{i}.h{j}' for i in range(2) for j in range(8))

def add_path_to_group(
    m: rc.IterativeMatcher,
    nodes_group: list[str],
    term_if_matches=True,
    qkv: Optional[str] = None,
):
    """Add the path from a matcher to a group of nodes using chain operation.

    If `term_if_matches=False` and `qkv` is not `None`, the `qkv` restrition will only be applied on the path to the first nodes found starting from `m`, indirect effect will not be restricted by `qkv`."""

    assert qkv in ['q', 'k', 'v', None]

    nodes_group: set = set(nodes_group)
    assert nodes_group.issubset(ALL_NODES_NAMES)
    nodes_to_ban = ALL_NODES_NAMES.difference(nodes_group)
    if qkv is None:
        attn_block_input = rc.new_traversal(start_depth=0, end_depth=1)
    else:
        attn_block_input = rc.restrict(f'a.{qkv}', term_if_matches=True,
                                       end_depth=8)

    return m.chain(attn_block_input).chain(
        rc.new_traversal(start_depth=1, end_depth=2).chain(
            rc.restrict(
                rc.Matcher(*nodes_group),
                term_early_at=rc.Matcher(nodes_to_ban),
                term_if_matches=term_if_matches,
            )
        )
    )

def make_matcher(
    match_nodes: list[str],
    term_if_matches=True,
    restrict=True,
    qkv: Optional[str] = None,
):
    m = rc.IterativeMatcher('final.call')
    if restrict:
        return add_path_to_group(m, match_nodes,
                                 term_if_matches=term_if_matches, qkv=qkv)
    else:
        return m.chain(rc.restrict(rc.Matcher(*match_nodes),
                                   term_if_matches=term_if_matches))


ind_matcher = make_matcher(['a1.h5', 'a1.h6'])
tot_scrub_loss = sample_and_evaluate(scrub_input(circuit, ind_matcher))

print(f'{tot_scrub_loss=}')

# %%

print_scrub = ScrubPrinter(toks_int_var.name, toks_int_var_other.name)

def solve_projection(
        c: rc.Circuit, matcher: rc.IterativeMatcher,
        proj_dim: Union[int, list[int]],
        batch_size=256, num_iters=250, adv_solving_mode=None,
        init_proj=None, pca=False, **kwargs
    ) -> Union[CircuitWithProjection, list[CircuitWithProjection]]:
    """solve the projection matrix by training on all samples

    :param matcher: matcher of the internal nodes that you care about
    :param kwargs: ``proj_args`` for :class:`LookAheadProjectionSolver`
    """
    init_torch_seed(92702102)       # for reproducibility

    logits_batch, logits_batch_in = expand_with_batch(
        c.get_unique('logits'), toks_int_var.name, batch_size)
    if isinstance(proj_dim, int):
        proj_dim = [proj_dim]
        return_single = True
    else:
        return_single = False

    solvers = [
        LookAheadProjectionSolver(
            logits_batch, matcher, proj_dim_i, logits_batch_in.name,
            optim_settings=rc.OptimizationSettings(
                max_memory=40*2**32, max_single_tensor_memory=40*2**32),
            proj_args=kwargs,
            use_pca_init=pca,
            adv_solving_mode=adv_solving_mode,
        )
        for proj_dim_i in proj_dim
    ]

    if init_proj is not None:
        assert len(solvers) == 1
        _, unexpected = solvers[0].proj.load_state_dict(init_proj.state_dict(),
                                                        strict=False)
        assert not unexpected

    dataset = toks_dataset

    for iter_num, (toks_batch, ), (toks_batch_other,) in zip(
            tqdm(range(num_iters), desc='solver proj'),
            iter_dataset_infinity(dataset, batch_size, True),
            iter_dataset_infinity(dataset, batch_size, True)):

        toks_batch = toks_batch.to(DEVICE)
        toks_batch_other = toks_batch_other.to(DEVICE)

        with torch.no_grad():
            seq_locs = good_induction_candidate[toks_batch[:, :-1]]

        # use the same shared_computing object in multiple solvers to avoid
        # repeated computation
        shared_computing = LookAheadProjectionSolver.SharedComputingContext()

        for i, solver in enumerate(solvers):
            solver.update(toks_batch, toks_batch_other, seq_locs,
                          shared_computing=shared_computing)
            if (iter_num % max(num_iters // 10, 1) == 0 or
                    iter_num == num_iters - 1):
                tqdm.write(
                    f'iter@{iter_num} solver#{i}: loss={solver.loss_mean:.5g} '
                    f'pmat={solver.proj.pmat}'
                )
        if all(i.suggest_early_stop for i in solvers):
            losses = ', '.join(f'{i.loss_mean:.5g}' for i in solvers)
            tqdm.write(f'follow early stop suggestion at iter {iter_num} with '
                       f'losses: {losses}')
            break

    ret = [
        i.replace_with_proj(
            c, lambda x: x.update(toks_int_var.name,
                                  lambda _: toks_int_var_other)
        )
        for i in solvers
    ]
    if return_single:
        return ret[0]
    return ret

# %%

# Helper functions for low-dim decomposition on the output of induction heads or
# on the output of prev-token head

ind_matcher = make_matcher(['a1.h6']).chain("a.attn_probs").chain("a.k").chain("a1.input")
def run_ind_head(title, need_ret=False, proj_dim=30, **kwargs):
    print(f'=============== projecting k-input of induction heads: {title}')
    p = solve_projection(circuit, ind_matcher, proj_dim=proj_dim, **kwargs)
    print_scrub(p.circuit)
    print(f'evaluating {p.proj.pmat}')
    loss = sample_and_evaluate(p.circuit)
    rate = (loss - tot_scrub_loss) / (baseline_loss - tot_scrub_loss )
    logprint(f'induction head: {title}: proj={p.proj}'
             f'{loss=:.4g} / ({baseline_loss:.4g}, {tot_scrub_loss:.4g}):'
             f' {rate*100:.2f}%')
    if need_ret:
        return p
# %%
for i in range(1, 33):
    proj = run_ind_head(f"dim{i}", need_ret=True, proj_dim=i)
    with open(f"data/projections/1.6k_pre_ln_{i}d.pkl", "wb") as f:
        pickle.dump(proj.proj, f)
# %%
