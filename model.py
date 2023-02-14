# %%
import torch
import utils
from functools import partial

import rust_circuit as rc
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id, negative_log_likelyhood
from interp.tools.indexer import SLICER as S
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR

from typing import Optional

DEVICE = "cuda:0"
SEQ_LEN = 300


@torch.inference_mode()
def load_model_and_data():
    model_id = "attention_only_2"
    (loaded, tokenizer, extra_args) = load_model_id(model_id)

    P = rc.Parser()
    toks_int_values = P("'toks_int_var' [104091,301] Array 3f36c4ca661798003df14994")
    toks_int_values = rc.cast_circuit(rc.Array(toks_int_values.value[:, :SEQ_LEN+1], name="toks_int_var"), rc.TorchDeviceDtypeOp(device=DEVICE, dtype="int64")).cast_array()
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
def construct_circuit(
    sub_all: bool = True,
    split_pth_ov_by_pt_or_not: bool = False,
    transpose_head=None,
    swap_q: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
    swap_k: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
    flip: Optional[tuple[int, int]] = None,
    pth_modify_only_children: list = [1, 2, 3],
    make_pth_true_prev: list = [],
    make_pth_beg_attend: list = [],
    make_pth_zero: list = [],
    actual_beg: int = 0,
    make_pth_diag: list = [],
    split_with_projection: list = [],
    split_paths_by_position: list = [],
):
    """Load the 2L attn-only model and make circuit that calculates loss on the dataset, with empty inputs"""

    orig_circuit, tok_embeds, pos_embeds, tokenizer, extra_args, toks_int_values = load_model_and_data()

    # sampling vars
    toks_int_var = rc.Array(torch.zeros(SEQ_LEN + 2, dtype=torch.int64).to(DEVICE), "toks_int_var")

    # input/expected
    input_toks = toks_int_var.index(I[:-2], name="input_toks_int")
    true_toks = toks_int_var.index(I[1:-1], name="true_toks_int")

    # feed input tokens to model (after embedding + causal mask)
    idxed_embeds = rc.GeneralFunction.gen_index(tok_embeds, input_toks, index_dim=0, name="idxed_embeds")
    causal_mask = rc.Array(
        (torch.arange(SEQ_LEN)[:, None] >= torch.arange(SEQ_LEN)[None, :]).to(tok_embeds.cast_array().value),
        f"t.a.c.causal_mask",
    )
    pos_embeds = pos_embeds.index(I[:SEQ_LEN], name="t.w.pos_embeds_idxed")
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

    by_head = configure_transformer(
        loss.get_unique("t.bind_w"),
        to=To.ATTN_HEAD_MLP_NORM,
        split_by_head_config={0: [(i, f"head{i}") for i in range(8)], 1: [(i, f"head{i}") for i in range(8)]},
        use_pull_up_head_split=True,
        check_valid=True,
    )

    # rename, conform
    by_head = by_head.update(lambda c: ".keep." in c.name, lambda c: c.rename(c.name.replace(".keep.", ".")))
    by_head = rc.conform_all_modules(by_head)

    expected_loss = loss.update("t.bind_w", lambda _: by_head)
    expected_loss = rc.conform_all_modules(expected_loss)

    model = (
        expected_loss.update("t.call", lambda c: c.cast_module().substitute())
        .update(rc.Regex(r".*?norm_call.*?"), lambda c: c.cast_module().substitute())
        .update(rc.Regex(r"\d\.a\."), lambda c: c.cast_module().substitute() if isinstance(c, rc.Module) else c)
    )

    if sub_all:
        model = rc.substitute_all_modules(model)
        #model.print_html(rc.PrintHtmlOptions(traversal=True))

    # FURTHER MODIFICATIONS

    # transpose a head (probably wrong)
    if transpose_head:
        model = model.update(
            rc.IterativeMatcher(transpose_head)
            .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
            .chain(rc.restrict("a.attn_scores_raw", end_depth=6)),
            lambda circ: rc.Einsum.from_einsum_string("rqk -> rkq" if len(circ.shape) == 3 else "qk -> kq", circ),
        )

    prev = ((torch.arange(SEQ_LEN)[:, None] - 1) == torch.arange(SEQ_LEN)[None, :]).to(tok_embeds.cast_array().value)
    prev[0, 0] = 1.0
    prev_mask = rc.Array(
        prev,
        "a.attn_probs",
    )

    for i in make_pth_true_prev:
        l1_head_matcher = rc.IterativeMatcher(f"b1.a.head{i}").children_matcher({0})
        model = model.update(
            l1_head_matcher.children_matcher(set(pth_modify_only_children))
            .chain("b0.a.head0")
            .chain("a.comb_v")
            .chain("a.attn_probs"),
            lambda c: prev_mask,
        )

    beg = (torch.arange(SEQ_LEN)[None, :] == (torch.zeros((SEQ_LEN, SEQ_LEN)) + actual_beg)).to(
        tok_embeds.cast_array().value
    )
    beg_mask = rc.Array(
        beg,
        "a.attn_probs",
    )
    for i in make_pth_beg_attend:
        l1_head_matcher = rc.IterativeMatcher(f"b1.a.head{i}").children_matcher({0})
        model = model.update(
            l1_head_matcher.children_matcher(set(pth_modify_only_children))
            .chain("b0.a.head0")
            .chain("a.comb_v")
            .chain("a.attn_probs"),
            lambda c: beg_mask,
        )

    zeros = torch.zeros((SEQ_LEN, SEQ_LEN)).to(tok_embeds.cast_array().value)
    zeros_mask = rc.Array(
        zeros,
        "a.attn_probs",
    )
    for i in make_pth_zero:
        l1_head_matcher = rc.IterativeMatcher(f"b1.a.head{i}").children_matcher({0})
        model = model.update(
            l1_head_matcher.children_matcher(set(pth_modify_only_children))
            .chain("b0.a.head0")
            .chain("a.comb_v")
            .chain("a.attn_probs"),
            lambda c: zeros_mask,
        )

    diag = ((torch.arange(SEQ_LEN)[:, None]) == torch.arange(SEQ_LEN)[None, :]).to(tok_embeds.cast_array().value)
    diag_mask = rc.Array(
        diag,
        "a.attn_probs",
    )

    for i in make_pth_diag:
        l1_head_matcher = rc.IterativeMatcher(f"b1.a.head{i}").children_matcher({0})
        model = model.update(
            l1_head_matcher.children_matcher(set(pth_modify_only_children)).chain("b0.a.head0").chain("a.comb_v").chain("a.attn_probs"),
            lambda c: diag_mask,
        )

    # scrub the ov by previous tokens or not
    if split_pth_ov_by_pt_or_not:
        k = rc.IterativeMatcher("a1.ind")

        # set up prev and not prev masks, will split v between these
        prev_mask = rc.Array(
            prev,
            "a.prev_tok_mask",
        )
        not_prev_mask = rc.Array(
            causal_mask.value - prev_mask.value,
            "a.not_prev_tok_mask",
        )

        # create new circuit in the comb_v position that splits up the value calculation
        attn_probs = model.get_unique(k.chain("a0.yes_prev").chain("a.comb_v").chain("a.attn_probs"))
        v = model.get_unique(k.chain("a0.yes_prev").chain("a.v"))
        new_comb_v = rc.Add(
            rc.Einsum.from_einsum_string(
                "qk,kV,qk->qV", attn_probs, v, prev_mask, name="a.attn_probs * a.prev_tok_mask"
            ),
            rc.Einsum.from_einsum_string(
                "qk,kV,qk->qV", attn_probs, v, not_prev_mask, name="a.attn_probs * a.not_prev_tok_mask'"
            ),
            name="a.comb_v",
        )

        model = model.update(k.chain("a0.yes_prev").chain("a.comb_v"), lambda x: new_comb_v)
        model = model.update(k.chain("a0.yes_prev").chain("a.head.on_inp"), lambda x: x.cast_module().substitute())

    # swapping weights between heads
    q = lambda l, h: rc.IterativeMatcher(f"b{l}.a.head{h}").chain(rc.restrict(f"a{l}.w.q.head{h}", end_depth=3))
    k = lambda l, h: rc.IterativeMatcher(f"b{l}.a.head{h}").chain(rc.restrict(f"a{l}.w.k.head{h}", end_depth=3))
    funcs = []
    swap = None
    if swap_q:
        funcs.append(q)
        swap = swap_q
    if swap_k:
        funcs.append(k)
        swap = swap_k

    if swap:
        for func in funcs:
            attn_probs1 = model.get_unique(func(*swap[0]))
            attn_probs2 = model.get_unique(func(*swap[1]))
            model = model.update(func(*swap[0]), lambda x: attn_probs2)
            model = model.update(func(*swap[1]), lambda x: attn_probs1)

    # transposing the attention scores of a particular head (correct)
    if flip:
        q_w = model.get_unique(q(*flip))
        k_w = model.get_unique(k(*flip))
        model = model.update(q(*flip), lambda x: k_w)
        model = model.update(k(*flip), lambda x: q_w)

    # split node with projection matrix
    for m, proj_name in split_with_projection:
        transform = partial(utils.split_circuit_with_projection, proj_name)
        model = model.update(m, lambda c: transform(c))

    # Split paths by position
    # This is so we can claim e.g. the value of 1.5 at position i depends solely on the ith token
    # The format is that for each l1 head we want to split some path for, we specify which child
    # (value, query, key), which l0 heads we want to split for, and how many of the most recent
    # tokens we want to include.
    # So for the above example, we should have (5, "v", [0, 1, 2, 3, 4, 5, 6, 7], 1).
    # To split for multiple children, add another tuple with the same l1 head.
    split_children = set()
    split_inputs = set()
    for l1h, child, l0hs, num_toks_back in split_paths_by_position:
        child_matcher = rc.restrict("b1").chain(rc.Matcher(rc.Regex(r"b1\.a\.head" + str(l1h)))).chain(rc.restrict(f"a.{child}", term_early_at="b0", term_if_matches=True))
        if (l1h, child) not in split_children:
            model = model.update(child_matcher.chain(rc.Matcher("b0")),
                                lambda c: rc.Concat(*[rc.Index(c, I[i:i+1], name=f"b0[{i}]") for i in range(SEQ_LEN)], axis=0, name="b0.concat")
            )
            split_children.add((l1h, child))
        for h in l0hs:
            assert (l1h, child, h) not in split_inputs
            for i in range(SEQ_LEN):
                input_matcher = child_matcher.chain(rc.Matcher(f"b0[{i}]")).chain(f"b0.a.head{h}").chain("input_toks_int")
                model = model.update(input_matcher, lambda c: rc.Concat(rc.Index(c, I[:i-(num_toks_back-1)], name="left_input_toks_int"),
                                                                        rc.Index(c, I[i-(num_toks_back-1):], name="right_input_toks_int"),
                                                                        axis=0, name="split_input_toks_int"
                ))
            split_inputs.add((l1h, child, h))

    return model, good_induction_candidate, tokenizer, toks_int_values


# %%
