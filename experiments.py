# %%
import argparse
from typing import Optional, Any
from functools import partial

import torch

import rust_circuit as rc
from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.causal_scrubbing.hypothesis import (
    CondSampler,
    Correspondence,
    ExactSampler,
    InterpNode,
    UncondSampler,
    corr_root_matcher,
)

from main import run_experiment
from model import construct_circuit


class FixedSampler(CondSampler):
    pos: int

    def __init__(self, pos=0):
        self.pos = pos

    def __call__(self, ref: Dataset, ds: Dataset, rng=None) -> Dataset:
        return ds[self.pos].sample(len(ref))


def make_corr(
    children: list[rc.IterativeMatcher] = [], options: Optional[dict[str, Any]] = None, sampler=ExactSampler()
):
    """Make a correspondence graph using a specific sampler."""
    corr = Correspondence()
    i_root = InterpNode(sampler, name="logits", other_inputs_sampler=sampler)
    corr.add(i_root, corr_root_matcher)
    for i, child in enumerate(children):
        tmp = i_root.make_descendant(UncondSampler(), name=f"{i}")
        corr.add(tmp, child)
    return (corr, options, children)


def m(*head: int):
    res = rc.IterativeMatcher(f"b0.a.head{head[0]}")
    for i in range(1, len(head)):
        res |= rc.IterativeMatcher(f"b0.a.head{head[i]}")
    return res


# EXPERIMENTS
def make_experiments(
    make_corr=make_corr,
) -> dict[str, tuple[Correspondence, dict[str, Optional[str]]]]:
    res = {}

    ind_heads = rc.Matcher("b1.a.head5") | rc.Matcher("b1.a.head6")
    non_ind_heads = rc.Matcher("b1.a.head0") | rc.Matcher("b1.a.head1") | rc.Matcher("b1.a.head2") | rc.Matcher("b1.a.head3") | rc.Matcher("b1.a.head4") | rc.Matcher("b1.a.head7")

    # shortcut matchers for useful parts of the graph
    embeds = rc.restrict("idxed_embeds", term_early_at="b0.a")
    ind_v = ind_heads.chain(rc.restrict(rc.Matcher("a.v"), term_early_at="b0"))
    ind_q = ind_heads.chain(rc.restrict(rc.Matcher("a.q"), term_early_at="b0"))
    ind_k = ind_heads.chain(rc.restrict(rc.Matcher("a.k"), term_early_at="b0"))

    # UNSCRUBBED
    res["unscrubbed"] = make_corr()

    res["scrub-ind-v"] = make_corr([ind_v])
    res["scrub-ind-q"] = make_corr([ind_q])
    res["scrub-ind-k"] = make_corr([ind_k])

    # head-k-child head
    for i in range(8):
        l1_head = rc.restrict(rc.Matcher(f"b1.a.head{i}"), term_if_matches=True)
        res[f"scrub-1.{i}"] = make_corr([l1_head])
        res[f"scrub-0.{i}"] = make_corr([m(i)])
        for ch in ["q", "k", "v"]:
            ch_matcher = l1_head.chain(rc.restrict(rc.Matcher(f"a.{ch}"), term_early_at="b0"))
            res[f"scrub-{ch}-1.{i}"] = make_corr([ch_matcher])
            res[f"scrub-{ch}-1.{i}-emb"] = make_corr([ch_matcher.chain(embeds)])
            for j in range(8):
                res[f"scrub-{ch}-1.{i}-0.{j}"] = make_corr([ch_matcher.chain(m(j))])
                res[f"scrub-{ch}-1.{i}-0.{j}e"] = make_corr([ch_matcher.chain(m(j) | embeds)])
            res[f"scrub-{ch}-1.{i}-0.06"] = make_corr([ch_matcher.chain(m(0, 6))])
            res[f"scrub-{ch}-1.{i}-0.47"] = make_corr([ch_matcher.chain(m(4, 7))])
            res[f"scrub-{ch}-1.{i}-0.0647"] = make_corr([ch_matcher.chain(m(0, 6, 4, 7))])
            res[f"scrub-{ch}-1.{i}-0.1235"] = make_corr([ch_matcher.chain(m(1, 2, 3, 5))])
            res[f"scrub-{ch}-1.{i}-0.1235e"] = make_corr([ch_matcher.chain(m(1, 2, 3, 5) | embeds)])
            res[f"scrub-{ch}-1.{i}-0.1235e-split"] = make_corr(
                [ch_matcher.chain(x) for x in [m(1), m(2), m(3), m(5), embeds]]
            )
            res[f"scrub-{ch}-1.{i}-0.123457"] = make_corr([ch_matcher.chain(m(1, 2, 3, 4, 5, 7))])
            res[f"scrub-{ch}-1.{i}-0.123457e"] = make_corr([ch_matcher.chain(m(1, 2, 3, 4, 5, 7) | embeds)])
            res[f"scrub-{ch}-1.{i}-0.1234567e"] = make_corr([ch_matcher.chain(m(1, 2, 3, 4, 5, 6, 7) | embeds)])
            res[f"scrub-{ch}-1.{i}-0.012356e"] = make_corr([ch_matcher.chain(m(0, 1, 2, 3, 5, 6) | embeds)])
            res[f"scrub-{ch}-1.{i}-0"] = make_corr([ch_matcher.chain(m(0, 1, 2, 3, 4, 5, 6, 7))])
        for j in range(8):
            res[f"scrub-1.{i}-0.{j}"] = make_corr([l1_head.chain(m(j))])

    # tranpose
    res["a1-ind-transpose"] = make_corr(options={"transpose_head": "a1.ind"})
    for l in range(2):
        for h in range(8):
            res[f"transpose-{l}{h}"] = make_corr(options={"split_heads": "all", "transpose_head": f"b{l}.a.head{h}"})
            res[f"transpose-{l}{h}-nopos"] = make_corr(options={"split_heads": "all", "flip": (l, h)})

    # swap
    for a in range(2 * 8):
        for b in range(a + 1, 2 * 8):
            l, h = a // 8, a % 8
            l2, h2 = b // 8, b % 8
            res[f"swap-{l}.{h}-{l2}.{h2}"] = make_corr(
                options={"split_heads": "all", "swap_k": ((l, h), (l2, h2)), "swap_q": ((l, h), (l2, h2))}
            )
            res[f"swap-k-{l}.{h}-{l2}.{h2}"] = make_corr(options={"split_heads": "all", "swap_k": ((l, h), (l2, h2))})
            res[f"swap-q-{l}.{h}-{l2}.{h2}"] = make_corr(options={"split_heads": "all", "swap_q": ((l, h), (l2, h2))})

    # BASELINE
    res["baseline"] = make_corr([ind_heads])
    res["not-baseline"] = make_corr([non_ind_heads])
    res["not-baseline-full"] = make_corr(
        [non_ind_heads | rc.IterativeMatcher(rc.restrict("b0", term_early_at="b1.a"))]
    )

    # EMBEDDING-VALUE
    ev = ind_v.chain("b0.a")
    res["ev"] = make_corr([ev])
    res["not-ev"] = make_corr([ind_v.chain(embeds)])

    # scrub each head individually
    for i in range(8):
        res[f"scrub-v-ind-0.{i}"] = make_corr([ind_v.chain(m(i))])
        res[f"scrub-v-ind-0.{i}e"] = make_corr([ind_v.chain(embeds | m(i))])

    # EMBEDDING-QUERY
    eq = ind_q.chain("b0.a")
    res["eq"] = make_corr([eq])
    res["not-eq"] = make_corr([ind_q.chain(embeds)])

    # scrub each head individually
    for i in range(8):
        res[f"scrub-q-ind-0.{i}"] = make_corr([ind_q.chain(m(i))])
        res[f"scrub-q-ind-0.{i}e"] = make_corr([ind_q.chain(embeds | m(i))])

    # EMBEDDING-KEY
    ek = ind_k.chain("b0.a")
    res["ek"] = make_corr([ek])
    res["not-ek"] = make_corr([ind_k.chain(embeds)])

    # PREVIOUS-TOKEN-HEAD KEY
    pth_k = ind_k.chain(embeds | m(1, 2, 3, 4, 5, 6, 7))
    res["pth-k"] = make_corr([pth_k])
    res["pth-k-full"] = make_corr(
        [
            pth_k
            | ind_k.chain("a.attn_probs * a.not_prev_tok_mask")
            | ind_k.chain("a.attn_probs * a.prev_tok_mask").chain("a.attn_probs")
        ],
        options={"split_pth_ov_by_pt_or_not": True},
    )

    # ALL (3 og scrubbing)
    res["all"] = make_corr([ev | eq | pth_k])
    for h in [5, 6]:
        ind_head = rc.IterativeMatcher(f"b1.a.head{h}").chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        res["all-1.6"] = make_corr(
            [
                ind_head.children_matcher({3}).chain("b0.a")
                | ind_head.children_matcher({1}).chain("b0.a")
                | ind_head.children_matcher({2}).chain(embeds | m(1, 2, 3, 4, 5, 6, 7)),
            ]
        )
        res["all-1.6-ind"] = make_corr(
            [ind_head.children_matcher({3}).chain(m(i)) for i in range(8)]
            + [ind_head.children_matcher({1}).chain(m(i)) for i in [0, 1, 2, 3, 4, 5, 7]]
            + [ind_head.children_matcher({2}).chain(embeds)]
            + [ind_head.children_matcher({2}).chain(m(i)) for i in [1, 2, 3, 4, 5, 7]]
        )

    res["positional-ev"] = make_corr(
        [ind_heads.chain(rc.Matcher("outside_input_toks_int"))],
        options={"split_paths_by_position": [
            (5, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
        ]}
    )

    res["positional-eq"] = make_corr(
        [ind_heads.chain(rc.Matcher("outside_input_toks_int"))],
        options={"split_paths_by_position": [
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
        ]}
    )

    res["positional-ek"] = make_corr(
        [ind_heads.chain(rc.Matcher("outside_input_toks_int"))],
        options={"split_paths_by_position": [
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
        ]}
    )

    res["pth-k+-with-positional-ek"] = make_corr(
        [ind_heads.chain(rc.Matcher("outside_input_toks_int"))],
        options={"split_paths_by_position": [
            (5, "k", [1, 2, 3, 4, 5, 7], 0, 1),
            (6, "k", [1, 2, 3, 4, 5, 7], 0, 1),
        ]}
    )

    res["pth-k+-with-positional-ek-no-47"] = make_corr(
        [
            (
                ind_heads.chain(rc.Matcher("outside_input_toks_int")) |
                ind_k.chain(rc.Regex("b0.a.head[47]"))
            ),
        ],
        options={"split_paths_by_position": [
            (5, "k", [1, 2, 3, 5], 0, 1),
            (6, "k", [1, 2, 3, 5], 0, 1),
        ]}
    )

    res["positional-pth-k+-with-positional-ek-no-47"] = make_corr(
        [
            (
                ind_heads.chain(rc.Matcher("outside_input_toks_int")) |
                ind_k.chain(rc.Regex("b0.a.head[47]"))
            ),
        ],
        options={"split_paths_by_position": [
            (5, "k", [1, 2, 3, 5], 0, 1),
            (5, "k", [0, 6], 2, 3),
            (6, "k", [1, 2, 3, 5], 0, 1),
            (6, "k", [0, 6], 2, 3),
        ]}
    )

    res["positional-pth-k+-with-positional-1.5-ek-no-47"] = make_corr(
        [
            (
                ind_heads.chain(rc.Matcher("outside_input_toks_int")) |
                rc.Matcher("b1.a.head5").chain(rc.restrict("a.k", term_early_at="b0")).chain(rc.Regex("b0.a.head[47]")) |
                rc.Matcher("b1.a.head6").chain(rc.restrict("a.k", term_early_at="b0")).chain(rc.Regex("b0.a.head[123457]"))
            ),
        ],
        options={"split_paths_by_position": [
            (5, "k", [1, 2, 3, 5], 0, 1),
            (5, "k", [0, 6], 2, 3),
            (6, "k", [0, 6], 2, 3),
        ]}
    )

    res["real-0.0"] = make_corr(options={"make_pth_true_prev": [0, 1, 2, 3, 4, 5, 6, 7]})
    res["beg-0.0"] = make_corr(options={"make_pth_beg_attend": [0, 1, 2, 3, 4, 5, 6, 7]})
    res["zero-0.0"] = make_corr(options={"make_pth_zero": [0, 1, 2, 3, 4, 5, 6, 7]})
    for ch, id in [("q", 1), ("k", 2), ("v", 3)]:
        res[f"{ch}-real-0.0"] = make_corr(
            options={
                "make_pth_true_prev": [0, 1, 2, 3, 4, 5, 6, 7],
                "pth_modify_only_children": [id],
            }
        )
        res[f"{ch}-beg-0.0"] = make_corr(
            options={
                "make_pth_beg_attend": [0, 1, 2, 3, 4, 5, 6, 7],
                "pth_modify_only_children": [id],
            }
        )
        res[f"{ch}-zero-0.0"] = make_corr(
            options={
                "make_pth_zero": [0, 1, 2, 3, 4, 5, 6, 7],
                "pth_modify_only_children": [id],
            }
        )
    for i in range(8):
        res[f"1.{i}-real-0.0"] = make_corr(options={"make_pth_true_prev": [i]})
        res[f"1.{i}-beg-0.0"] = make_corr(options={"make_pth_beg_attend": [i]})
        res[f"1.{i}-pos20-0.0"] = make_corr(options={"make_pth_beg_attend": [i], "actual_beg": 20})
        res[f"1.{i}-pos1-0.0"] = make_corr(options={"make_pth_beg_attend": [i], "actual_beg": 1})
        res[f"1.{i}-zero-0.0"] = make_corr(options={"make_pth_zero": [i]})
        res[f"1.{i}-diag-0.0"] = make_corr(options={"make_pth_diag": [i]})
        for ch, id in [("q", 1), ("k", 2), ("v", 3)]:
            res[f"{ch}-1.{i}-real-0.0"] = make_corr(
                options={
                    "make_pth_true_prev": [i],
                    "pth_modify_only_children": [id],
                }
            )
            res[f"{ch}-1.{i}-beg-0.0"] = make_corr(
                options={
                    "make_pth_beg_attend": [i],
                    "pth_modify_only_children": [id],
                }
            )
            res[f"{ch}-1.{i}-zero-0.0"] = make_corr(
                options={
                    "make_pth_zero": [i],
                    "pth_modify_only_children": [id],
                }
            )
            res[f"{ch}-1.{i}-diag-0.0"] = make_corr(
                options={
                    "make_pth_diag": [i],
                    "pth_modify_only_children": [id],
                }
            )

    for i in range(1, 256):
        res[f"k-1.6-pre-ln-proj-{i}d"] = make_corr(
            [rc.Matcher("proj_residual")],
            options={
                "split_with_projection": [
                    (
                        rc.IterativeMatcher("b1").chain(
                            rc.restrict("b1.a.head6", end_depth=3)).chain(
                            rc.restrict("a.attn_probs", end_depth=3)).chain(
                            rc.restrict("a.k", end_depth=4)).chain(
                            rc.restrict("b0")
                        ), f"1.6k_pre_ln_{i}d"),
                ]
            }
        )

    for i in range(1, 256):
        res[f"k-1.6-post-ln-proj-{i}d"] = make_corr(
            [rc.Matcher("proj_residual")],
            options={
                "split_with_projection": [
                    (
                        rc.IterativeMatcher("b1").chain(
                            rc.restrict("b1.a.head6", end_depth=3)).chain(
                            rc.restrict("a.attn_probs", end_depth=3)).chain(
                            rc.restrict("a.k", end_depth=4)).chain(
                            rc.restrict("a1.norm", end_depth=3)
                        ), f"1.6k_post_ln_{i}d"),
                ]
            }
        )

    for i in range(1, 256):
        res[f"k-1.6-l0-out-proj-{i}d"] = make_corr(
            [rc.Matcher("proj_residual")],
            options={
                "split_with_projection": [
                    (
                        rc.IterativeMatcher("b1").chain(
                            rc.restrict("b1.a.head6", end_depth=3)).chain(
                            rc.restrict("a.attn_probs", end_depth=3)).chain(
                            rc.restrict("a.k", end_depth=4)).chain(
                            rc.restrict("b0")).chain(
                            rc.restrict("b0.a")
                        ), f"1.6k_l0_out_{i}d"),
                ]
            }
        )

    for i in range(1, 256):
        res[f"k-1.6-l0h0-out-proj-{i}d"] = make_corr(
            [rc.Matcher("proj_residual")],
            options={
                "split_with_projection": [
                    (
                        rc.IterativeMatcher("b1").chain(
                            rc.restrict("b1.a.head6", end_depth=3)).chain(
                            rc.restrict("a.attn_probs", end_depth=3)).chain(
                            rc.restrict("a.k", end_depth=4)).chain(
                            rc.restrict("b0")).chain(
                            rc.restrict("b0.a")).chain(
                            rc.restrict("b0.a.head0")
                        ), f"1.6k_l0h0_out_{i}d"),
                ]
            }
        )

    for i in range(1, 256):
        res[f"k-1.6-l0h6-out-proj-{i}d"] = make_corr(
            [rc.Matcher("proj_residual")],
            options={
                "split_with_projection": [
                    (
                        rc.IterativeMatcher("b1").chain(
                            rc.restrict("b1.a.head6", end_depth=3)).chain(
                            rc.restrict("a.attn_probs", end_depth=3)).chain(
                            rc.restrict("a.k", end_depth=4)).chain(
                            rc.restrict("b0")).chain(
                            rc.restrict("b0.a")).chain(
                            rc.restrict("b0.a.head6")
                        ), f"1.6k_l0h6_out_{i}d"),
                ]
            }
        )


    for i in range(1, 256):
        res[f"k-1.6-l0h06-out-proj-{i}d"] = make_corr(
            [rc.Matcher("proj_residual")],
            options={
                "split_with_projection": [
                    (
                        rc.IterativeMatcher("b1").chain(
                            rc.restrict("b1.a.head6", end_depth=3)).chain(
                            rc.restrict("a.attn_probs", end_depth=3)).chain(
                            rc.restrict("a.k", end_depth=4)).chain(
                            rc.restrict("b0")).chain(
                            rc.restrict("b0.a")).chain(
                            rc.restrict(rc.Matcher("b0.a.head0") | rc.Matcher("b0.a.head6"))
                        ), f"1.6k_l0h06_out_{i}d"),
                ]
            }
        )

    for i in range(1, 256):
        res[f"k-1.5-transplant-1.6-l0-out-proj-{i}d"] = make_corr(
            [rc.Matcher("proj_residual")],
            options={
                "split_with_projection": [
                    (
                        rc.IterativeMatcher("b1").chain(
                            rc.restrict("b1.a.head5", end_depth=3)).chain(
                            rc.restrict("a.attn_probs", end_depth=3)).chain(
                            rc.restrict("a.k", end_depth=4)).chain(
                            rc.restrict("b0")).chain(
                            rc.restrict("b0.a")
                        ), f"1.6k_l0_out_{i}d"),
                ]
            }
        )

    for i in range(1, 256):
        res[f"k-1.5-transplant-1.6-l0h0-out-proj-{i}d"] = make_corr(
            [rc.Matcher("proj_residual")],
            options={
                "split_with_projection": [
                    (
                        rc.IterativeMatcher("b1").chain(
                            rc.restrict("b1.a.head5", end_depth=3)).chain(
                            rc.restrict("a.attn_probs", end_depth=3)).chain(
                            rc.restrict("a.k", end_depth=4)).chain(
                            rc.restrict("b0")).chain(
                            rc.restrict("b0.a")).chain(
                            rc.restrict("b0.a.head0")
                        ), f"1.6k_l0h0_out_{i}d"),
                ]
            }
        )
    return res


# %%
def main():
    experiments = make_experiments()

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run causal scrubbing experiments on induction heads in the 2L attn-only model"
    )
    parser.add_argument(
        "--exp", action="store", dest="exp_name", type=str, choices=list(experiments.keys()), default="unscrubbed"
    )
    parser.add_argument("--samples", action="store", dest="samples", type=int, default=10000)
    parser.add_argument("--verbose", action="store", dest="verbose", type=int, default=0)
    parser.add_argument("--attns", action="store_true", dest="attns")
    parser.add_argument("--attn-scores", action="store_true", dest="attn_scores")
    parser.add_argument("--save", action="store_true", dest="save")
    parser.add_argument("--positional-scrub", action="store_true", dest="positional_scrub")
    parser.add_argument(
        "--idx",
        action="store",
        dest="idx",
        type=int,
        default=None,
        help="Dataset index for subgraph ablation attribution",
    )
    args = parser.parse_args()
    print(args)

    save_name = ""
    if args.save:
        save_name = args.exp_name
    if args.idx is not None:
        experiments = make_experiments(partial(make_corr, sampler=FixedSampler(args.idx)))
        if args.save:
            if args.attns:
                save_name += f"_attns_{args.idx}"
            else:
                save_name += f"_saa_{args.idx}"

    run_experiment(experiments, args.exp_name, args.samples, save_name, args.verbose, args.attns, args.attn_scores, args.positional_scrub)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
