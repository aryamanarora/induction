import torch

import rust_circuit as rc
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id, negative_log_likelyhood
from interp.tools.indexer import SLICER as S
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR

DEVICE = "cuda:0"
seq_len = 300

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

@torch.inference_mode()
def load_model_and_data():
    model_id = "attention_only_2"
    (loaded, tokenizer, extra_args) = load_model_id(model_id)

    P = rc.Parser()
    toks_int_values = P("'toks_int_var' [104091,301] Array 3f36c4ca661798003df14994")
    toks_int_values = rc.cast_circuit(
        toks_int_values, rc.TorchDeviceDtypeOp(device=DEVICE, dtype="int64")
    ).cast_array()
    toks_indices = torch.arange(toks_int_values.shape[0], device=DEVICE).reshape(-1, 1)
    toks_int_values = rc.Array(
        torch.concat([toks_int_values.cast_array().value, toks_indices], dim=1),
        name="toks_int_var",
    )
    loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE)) for (s, c) in loaded.items()}

    orig_circuit = loaded["t.bind_w"]
    tok_embeds = loaded["t.w.tok_embeds"]
    pos_embeds = loaded["t.w.pos_embeds"]

    return orig_circuit, tok_embeds, pos_embeds, tokenizer, extra_args, toks_int_values


@torch.inference_mode()
def construct_circuit(split_heads: str = "labelled"):
    """Load the 2L attn-only model and make circuit that calculates loss on the dataset, with empty inputs"""

    orig_circuit, tok_embeds, pos_embeds, tokenizer, extra_args, toks_int_values = load_model_and_data()

    # sampling vars
    toks_int_var = rc.Array(torch.zeros(302, dtype=torch.int64).to(DEVICE), "toks_int_var")

    # input/expected
    input_toks = toks_int_var.index(I[:-2], name="input_toks_int")
    true_toks = toks_int_var.index(I[1:-1], name="true_toks_int")

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

    # split by heads
    assert split_heads in list(split_head_configs.keys())
    split_head_config = split_head_configs[split_heads]

    by_head = configure_transformer(
        loss.get_unique("t.bind_w"),
        to=To.ATTN_HEAD_MLP_NORM,
        split_by_head_config=split_head_config,
        use_pull_up_head_split=True,
        check_valid=True,
    )

    # rename, conform
    by_head = by_head.update(lambda c: ".keep." in c.name, lambda c: c.rename(c.name.replace(".keep.", ".")))
    by_head = rc.conform_all_modules(by_head)

    expected_loss = loss.update("t.bind_w", lambda _: by_head)
    expected_loss = rc.conform_all_modules(expected_loss)

    with_a1_ind_inputs = (
        expected_loss.update("a1.ind_sum.norm_call", lambda c: c.cast_module().substitute())
        .update("b1.a.ind_sum", lambda c: c.cast_module().substitute())
        .update("t.call", lambda c: c.cast_module().substitute())
        .update("a1.not_ind_sum.norm_call", lambda c: c.cast_module().substitute())
        .update("b1.a.not_ind_sum", lambda c: c.cast_module().substitute())
    )
    return with_a1_ind_inputs, good_induction_candidate, tokenizer, toks_int_values