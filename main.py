# %%
import os
import subprocess
import sys
import uuid
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

MAIN = __name__ == "__main__"
DEVICE = "cuda:0"
FILTER_INDUCT = False
seq_len = 300
n_files = 12

# %%
@torch.inference_mode()
def load_model_and_data():
    model_id = "attention_only_2"
    (loaded, tokenizer, extra_args) = load_model_id(model_id)

    toks_int_values: rc.Array
    P = rc.Parser()
    toks_int_values = P("'toks_int_var' [104091,301] Array 3f36c4ca661798003df14994").as_array_unwrap()
    toks_int_values = rc.cast_circuit(
        toks_int_values, rc.TorchDeviceDtypeOp(device=DEVICE, dtype="int64")
    ).as_array_unwrap()
    loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE)) for (s, c) in loaded.items()}

    orig_circuit = loaded["t.bind_w"]
    tok_embeds = loaded["t.w.tok_embeds"]
    pos_embeds = loaded["t.w.pos_embeds"]

    return orig_circuit, tok_embeds, pos_embeds, tokenizer, extra_args, toks_int_values


# %%
orig_circuit, tok_embeds, pos_embeds, tokenizer, extra_args, toks_int_values = load_model_and_data()

# sampling vars
toks_int_var = rc.Array(torch.zeros(301, dtype=torch.int64).to(DEVICE), "toks_int_var")

# input/expected
input_toks = toks_int_var.index(I[:-1], name="input_toks_int")
true_toks = toks_int_var.index(I[1:], name="true_toks_int")

# feed input tokens to model (after embedding + causal mask)
idxed_embeds = rc.GeneralFunction.gen_index(tok_embeds, input_toks, index_dim=0, name="idxed_embeds")
causal_mask = rc.Array(
    (torch.arange(seq_len)[:, None] >= torch.arange(seq_len)[None, :]).to(tok_embeds.as_array_unwrap().value),
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
is_good_induction_candidate = rc.GeneralFunction.gen_index(
    x=rc.Array(good_induction_candidate, name="tok_is_induct_candidate"),
    index=input_toks,
    index_dim=0,
    name="induct_candidate",
)

# mean loss only on induction candidates
if FILTER_INDUCT:
    loss = rc.Einsum((loss, (0,)), (is_good_induction_candidate, (0,)), out_axes=(0,), name="loss_on_candidates")
expected_loss_by_seq = rc.Cumulant(loss, name="t.expected_loss_by_seq")
expected_loss_old = expected_loss_by_seq.mean(name="t.expected_loss", scalar_name="recip_seq")

# helpers for sampling and evaluating
# def seeder(c: rc.Circuit) -> int:
#     if c == toks_int_var.probs_and_group:
#         return 11
#     elif c == toks_int_var_other.probs_and_group:
#         return 22
#     else:
#         raise ValueError("Expected one of the probs_and_group we constructed earlier, but got something else!", c)


# def sample_and_evaluate(c: rc.Circuit, num_samples: int = 16 * 128, batch_size=32) -> float:
#     def run_on_sampled(c: rc.Circuit) -> rc.Circuit:
#         return rc.batch_to_concat(rc.substitute_all_modules(c), axis=0, num_batches=num_samples // batch_size)

#     sampler = rc.Sampler(
#         rc.RandomSampleSpec((num_samples,), seeder=seeder), run_on_sampled=run_on_sampled, device_dtype=c.device_dtype
#     )
#     return rc.optimize_and_evaluate(sampler.estimate(c)).item()


# sample_and_evaluate(expected_loss_old)
###########################################################################################################
# %%
# Color printing by input

# scrubbed = lambda c: c.are_any_found(toks_int_var_other)
# not_scrubbed = lambda c: c.are_any_found(toks_int_var)


# def scrub_colorer(c):
#     getting_scrubbed = c.are_any_found(toks_int_var_other)
#     getting_unscrubbed = c.are_any_found(toks_int_var)
#     if getting_scrubbed and getting_unscrubbed:
#         return "purple"
#     elif getting_scrubbed:
#         return "red"
#     elif getting_unscrubbed:
#         return "cyan"
#     else:
#         return "lightgrey"


# printer = rc.PrintHtmlOptions(colorer=scrub_colorer, traversal=True)

# expected_loss_old.print(printer)
# %%
# Split by heads, rename, conform
@torch.inference_mode()
def clean_model(expected_loss_old: rc.Circuit):
    by_head = configure_transformer(
        expected_loss_old.get_unique("t.bind_w"),
        to=To.ATTN_HEAD_MLP_NORM,
        split_by_head_config={
            0: [(0, "prev"), (S[1:], "not_prev")],
            1: [
                (S[5:7], "ind"),
                (torch.tensor([0, 1, 2, 3, 4, 7]).to(model.device), "not_ind"),
            ],
        },
        use_pull_up_head_split=True,
        check_valid=True,
    )

    by_head = by_head.update(lambda c: ".keep." in c.name, lambda c: c.rename(c.name.replace(".keep.", ".")))
    by_head = rc.conform_all_modules(by_head)

    expected_loss = expected_loss_old.update("t.bind_w", lambda _: by_head)
    expected_loss = rc.conform_all_modules(expected_loss)

    with_a1_ind_inputs = (
        expected_loss.update("a1.ind_sum.norm_call", lambda c: c.as_module_unwrap().substitute())
        .update("b1.a.ind_sum", lambda c: c.as_module_unwrap().substitute())
        .update("t.call", lambda c: c.as_module_unwrap().substitute())
    )
    with_a1_ind_inputs.print_html()
    return with_a1_ind_inputs


with_a1_ind_inputs = clean_model(loss)
# %%
corr = Correspondence()

i_root = InterpNode(ExactSampler(), name="logits")
a1_ind = i_root.make_descendant(ExactSampler(), name="a1.ind")
i_root.print()

m_root = corr_root_matcher
corr.add(i_root, m_root)
corr.add(a1_ind, rc.IterativeMatcher("a1.ind"))

# %%
with torch.inference_mode():
    ds = Dataset({"toks_int_var": toks_int_values})
    eval_settings = ExperimentEvalSettings(device_dtype=DEVICE, run_on_all=False)
    exp = Experiment(with_a1_ind_inputs, ds, corr, 100, random_seed=42)
    print(exp.sample(exp.make_ref_ds())[i_root])
    scrubbed_circuit = exp.scrub()
    # sampler = eval_settings.get_sampler(len(scrubbed_circuit.ref_ds), None)
    # print(sampler.estimate_and_sample(scrubbed_circuit.circuit).evaluate())
    scrubbed_circuit.print()
    # scrubbed_circuit.circuit.update(True, lambda x: (x, print(x, x.is_explicitly_computable) if not x.is_explicitly_computable else '')[0])
    # print(scrubbed_circuit.circuit.print_html())
    res = scrubbed_circuit.evaluate(eval_settings)
    print(res[res != 0.0].mean())
    print(res.mean())
    print(res)
# %%
with_a1_ind_inputs.print_html()
# %%
