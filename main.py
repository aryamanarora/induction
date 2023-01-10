import os
import subprocess
import sys
import uuid
import pickle
from pprint import pprint
import rust_circuit as rc
import torch
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id, negative_log_likelyhood
from interp.tools.data_loading import get_val_seqs
from interp.tools.indexer import SLICER as S
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR
from interp.circuit.causal_scrubbing.hypothesis import (
    CondSampler,
    Correspondence,
    ExactSampler,
    InterpNode,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
from interp.circuit.causal_scrubbing.testing_utils import IntDataset
from interp.circuit.causal_scrubbing.experiment import Experiment, ExperimentEvalSettings, ScrubbedExperiment
from interp.circuit.causal_scrubbing.dataset import color_dataset, Dataset
from torch.testing import assert_close
from collections import defaultdict
from tqdm import tqdm

MAIN = __name__ == "__main__"
DEVICE = "cuda:0"
FILTER_INDUCT = True
seq_len = 300
n_files = 12


@torch.inference_mode()
def construct_circuit():
    """Load the 2L attn-only model and make circuit that calculates loss on the dataset, with empty inputs"""

    @torch.inference_mode()
    def load_model_and_data():
        model_id = "attention_only_2"
        (loaded, tokenizer, extra_args) = load_model_id(model_id)

        P = rc.Parser()
        toks_int_values = P("'toks_int_var' [104091,301] Array 3f36c4ca661798003df14994")
        toks_int_values = rc.cast_circuit(
            toks_int_values, rc.TorchDeviceDtypeOp(device=DEVICE, dtype="int64")
        ).cast_array()
        loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE)) for (s, c) in loaded.items()}

        orig_circuit = loaded["t.bind_w"]
        tok_embeds = loaded["t.w.tok_embeds"]
        pos_embeds = loaded["t.w.pos_embeds"]

        return orig_circuit, tok_embeds, pos_embeds, tokenizer, extra_args, toks_int_values

    orig_circuit, tok_embeds, pos_embeds, tokenizer, extra_args, toks_int_values = load_model_and_data()

    # sampling vars
    toks_int_var = rc.Array(torch.zeros(301, dtype=torch.int64).to(DEVICE), "toks_int_var")

    # input/expected
    input_toks = toks_int_var.index(I[:-1], name="input_toks_int")
    true_toks = toks_int_var.index(I[1:], name="true_toks_int")

    # feed input tokens to model (after embedding + causal mask)
    idxed_embeds = rc.GeneralFunction.gen_index(tok_embeds, input_toks, index_dim=0, name="idxed_embeds")
    causal_mask = rc.Array(
        (torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]).to(tok_embeds.cast_array().value),
        f"t.a.c.causal_mask",
    )
    pos_embeds = pos_embeds.index(I[:seq_len], name="t.w.pos_embeds_idxed")
    model = rc.module_new_bind(
        orig_circuit, ("t.input", idxed_embeds), ("a.mask", causal_mask), ("a.pos_input", pos_embeds), name="t.call"
    )

    # final loss circuit
    loss = rc.Module(negative_log_likelyhood.spec, **{"ll.input": model, "ll.label": true_toks}, name="t.loss")

    # load good induction candidate data
    CACHE_DIR = f"{RRFS_DIR}/ryan/induction_scrub/cached_vals"
    good_induction_candidate = torch.load(f"{CACHE_DIR}/induction_candidates_2022-10-15 04:48:29.970735.pt").to(
        device=DEVICE, dtype=torch.float32
    )

    return loss, good_induction_candidate, tokenizer, toks_int_values


def get_induction_candidate_masks(
    t: torch.Tensor, good_induction_candidates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    t is a 2d Tensor of token indices of size batch_size x seq_len
    good_induction_candidate is a 1d Tensor of 0s and 1s of size vocab_size indicating whether the ith token is a good induction candidate in general
    Return two 2d Tensors of bools indicating whether each token in t is a good induction candidate in that row (in the second tensor, we exclude first occurrences of each token in each row)
    """
    res_all = torch.zeros_like(t, dtype=torch.bool)
    res_later = torch.zeros_like(t, dtype=torch.bool)
    good_induction_candidates = good_induction_candidates.to(dtype=torch.bool)
    # Sorry, couldn't find anything better than a double-for
    for i, row in tqdm(enumerate(t), total=t.shape[0]):
        seen_toks = set()
        for j, tok in enumerate(row):
            if good_induction_candidates[tok]:
                res_all[i, j] = True
                if tok.item() in seen_toks:
                    res_later[i, j] = True
                seen_toks.add(tok.item())

    return res_all, res_later


split_head_configs = {
    "labelled": {
        0: [(0, "yes_prev"), (S[1:], "not_prev")],
        1: [
            (S[5:7], "ind"),
            (torch.tensor([0, 1, 2, 3, 4, 7]).to(DEVICE), "not_ind"),
        ],
    },
    "all": {0: [(i, f"head{i}") for i in range(8)], 1: [(i, f"head{i}") for i in range(8)]},
    "b0-all": {
        0: [(i, f"head{i}") for i in range(8)],
        1: [
            (S[5:7], "ind"),
            (torch.tensor([0, 1, 2, 3, 4, 7]).to(DEVICE), "not_ind"),
        ],
    },
}

# Split by heads, rename, conform
@torch.inference_mode()
def clean_model(expected_loss_old: rc.Circuit, split_heads: str = "labelled"):
    assert split_heads in list(split_head_configs.keys())
    split_head_config = split_head_configs[split_heads]

    by_head = configure_transformer(
        expected_loss_old.get_unique("t.bind_w"),
        to=To.ATTN_HEAD_MLP_NORM,
        split_by_head_config=split_head_config,
        use_pull_up_head_split=True,
        check_valid=True,
    )

    by_head = by_head.update(lambda c: ".keep." in c.name, lambda c: c.rename(c.name.replace(".keep.", ".")))
    by_head = rc.conform_all_modules(by_head)

    expected_loss = expected_loss_old.update("t.bind_w", lambda _: by_head)
    expected_loss = rc.conform_all_modules(expected_loss)

    with_a1_ind_inputs = (
        expected_loss.update("a1.ind_sum.norm_call", lambda c: c.cast_module().substitute())
        .update("b1.a.ind_sum", lambda c: c.cast_module().substitute())
        .update("t.call", lambda c: c.cast_module().substitute())
        .update("a1.not_ind_sum.norm_call", lambda c: c.cast_module().substitute())
        .update("b1.a.not_ind_sum", lambda c: c.cast_module().substitute())
    )
    return with_a1_ind_inputs


@torch.inference_mode()
def run_hypothesis(
    circuit: rc.Circuit,
    toks: rc.Array,
    correspondence: Correspondence,
    good_induction_candidates,
    samples=10000,
    tokenizer=None,
    verbose=0,
    seed: int = 42,
    save_name="",
):
    if verbose:
        print("Running hypothesis")
    ds = Dataset({"toks_int_var": toks})
    eval_settings = ExperimentEvalSettings(device_dtype=DEVICE, batch_size=100, run_on_all=True)

    exp = Experiment(circuit, ds, correspondence, num_examples=samples, random_seed=seed)
    scrubbed_circuit = exp.scrub()
    inps = get_inputs_from_model(scrubbed_circuit.circuit)
    res = scrubbed_circuit.evaluate(eval_settings)
    if verbose == 2:
        scrubbed_circuit.print()
        if tokenizer is not None:
            pprint(tokenizer.batch_decode(inps))
            binps = inps.clone()
            binps[get_induction_candidate_masks(inps, good_induction_candidates)[0]] = inps[0][0]
            pprint(tokenizer.batch_decode(binps))

    if verbose:
        print("Building induction candidates masks")
    ind_candidates_mask, ind_candidates_later_occur_mask = get_induction_candidate_masks(
        inps[:, :-1], good_induction_candidates
    )

    if save_name:
        with open(f"data/{save_name}.pkl", "wb") as f:
            pickle.dump((res, ind_candidates_mask, ind_candidates_later_occur_mask), f)
        with open(f"data/inps_{save_name}.pkl", "wb") as f:
            pickle.dump(inps, f)
    return res, ind_candidates_mask, ind_candidates_later_occur_mask, scrubbed_circuit


def get_inputs_from_model(model: rc.Circuit):
    data = model.get_unique("true_toks_int").get_unique("toks_int_var")
    return data.evaluate()


def run_experiment(
    exps, exp_name: str, model: rc.Circuit, toks, candidates, tokenizer, samples=10000, save_results=False, verbose=0
):
    mean_overall_losses = defaultdict(lambda: torch.zeros(2))
    save_name = f"{exp_name}" if save_results else ""
    res, ind_candidates_mask, ind_candidates_later_occur_mask, scrubbed_circuit = run_hypothesis(
        model, toks, exps[exp_name][0], candidates, samples=samples, tokenizer=tokenizer, verbose=verbose, seed=42, save_name=save_name
    )
    print(exp_name.upper())
    print("OVERALL")
    print(f"{res.mean().item():>10.3f}{res.var().item():>10.3f}{res.shape[0] * res.shape[1]:>10}")

    print("CANDIDATES")
    c_res = res[ind_candidates_mask]
    print(f"{c_res.mean().item():>10.3f}{c_res.var().item():>10.3f}{c_res.shape[0]:>10}")
    # print(f"unnormed {(res * ind_candidates_mask).mean():.3f}")

    print("LATER CANDIDATES")
    lc_res = res[ind_candidates_later_occur_mask]
    print(f"{lc_res.mean().item():>10.3f}{lc_res.var().item():>10.3f}{lc_res.shape[0]:>10}")