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

    return loss, good_induction_candidate, tokenizer, toks_int_values


def get_induction_candidate_mask(
    t: torch.Tensor, good_induction_candidates: torch.Tensor, match_all_occurrences=False
) -> torch.Tensor:
    """
    t is a 2d Tensor of token indices of size batch_size x seq_len
    good_induction_candidate is a 1d Tensor of 0s and 1s of size vocab_size indicating whether the ith token is a good induction candidate in general
    Return a 2d Tensor of bools indicating whether each token in t is a repeated occurrence of a good induction candidate in that row (or, if match_all_occurrences is True, we also set the first occurrence of the token to True)
    """
    res = torch.ones_like(t, dtype=torch.bool)
    good_induction_candidates = good_induction_candidates.to(dtype=torch.bool)
    # Sorry, couldn't find anything better than a double-for
    for i, row in enumerate(t):
        seen_toks = set()
        for j, tok in enumerate(row):
            if tok.item() in seen_toks or match_all_occurrences:
                res[i, j] = good_induction_candidates[tok]
            else:
                seen_toks.add(tok.item())
                res[i, j] = False

    return res


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
                (torch.tensor([0, 1, 2, 3, 4, 7]).to(DEVICE), "not_ind"),
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
        .update("a1.not_ind_sum.norm_call", lambda c: c.as_module_unwrap().substitute())
        .update("b1.a.not_ind_sum", lambda c: c.as_module_unwrap().substitute())
    )
    with_a1_ind_inputs.print_html()
    return with_a1_ind_inputs


@torch.inference_mode()
def run_hypothesis(
    circuit: rc.Circuit,
    toks: rc.Array,
    correspondence: Correspondence,
    good_induction_candidates,
    samples=100,
    tokenizer=None,
    p=False,
):
    ds = Dataset({"toks_int_var": toks})
    eval_settings = ExperimentEvalSettings(device_dtype=DEVICE, run_on_all=False)
    exp = Experiment(circuit, ds, correspondence, samples, random_seed=42)
    scrubbed_circuit = exp.scrub()
    if p:
        scrubbed_circuit.print()

    inps = get_inputs_from_model(scrubbed_circuit.circuit)
    res = scrubbed_circuit.evaluate(eval_settings)
    overall_mean_loss = res.mean()
    ind_candidates_mask = get_induction_candidate_mask(
        inps[:, :-1], good_induction_candidates, match_all_occurrences=True
    )
    if tokenizer is not None and p:
        pprint(tokenizer.batch_decode(inps))
        binps = inps.clone()
        binps[get_induction_candidate_mask(inps, good_induction_candidates, match_all_occurrences=True)] = inps[0][0]
        pprint(tokenizer.batch_decode(binps))

    ind_candidates_mean_loss = (res[ind_candidates_mask]).mean()
    ind_candidates_later_occurr_mask = get_induction_candidate_mask(
        inps[:, :-1], good_induction_candidates, match_all_occurrences=False
    )
    ind_candidates_later_occurr_mean_loss = (res[ind_candidates_later_occurr_mask]).mean()
    mean_losses = {
        "overall": (overall_mean_loss.item(), (res.shape[0] * res.shape[1])),
        "candidates_all": (ind_candidates_mean_loss.item(), ind_candidates_mask.sum().item()),
        "candidates_later": (
            ind_candidates_later_occurr_mean_loss.item(),
            ind_candidates_later_occurr_mask.sum().item(),
        ),
    }
    return res, scrubbed_circuit, mean_losses


def get_inputs_from_model(model: rc.Circuit):
    data = model.get_unique("true_toks_int").get_unique("toks_int_var")
    return data.evaluate()


loss, good_induction_candidate, tokenizer, toks_int_values = construct_circuit()
with_a1_ind_inputs = clean_model(loss)

# UNSCRUBBED
corr = Correspondence()
i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
corr.add(i_root, corr_root_matcher)

res, scrubbed_circuit, mean_losses = run_hypothesis(
    with_a1_ind_inputs, toks_int_values, corr, good_induction_candidate, tokenizer=tokenizer, p=False
)
tokens = get_inputs_from_model(scrubbed_circuit.circuit)

print("UNSCRUBBED")
pprint(mean_losses)

# BASELINE
a1_ind = i_root.make_descendant(UncondSampler(), name="a1.ind")
corr.add(a1_ind, rc.IterativeMatcher("a1.ind"))

res, scrubbed_circuit, mean_losses = run_hypothesis(with_a1_ind_inputs, toks_int_values, corr, good_induction_candidate)
print("BASELINE")
pprint(mean_losses)

#

torch.cuda.empty_cache()

# %%
