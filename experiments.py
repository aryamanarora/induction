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

    # BASELINE
    res["baseline"] = make_corr([ind_heads])
    res["baseline-decohered-by-child"] = make_corr([ind_v, ind_q, ind_k])
    res["baseline-decohered-by-head-and-child"] = make_corr([
        rc.Matcher("b1.a.head5").chain(rc.restrict("a.v", term_early_at="b0")),
        rc.Matcher("b1.a.head5").chain(rc.restrict("a.q", term_early_at="b0")),
        rc.Matcher("b1.a.head5").chain(rc.restrict("a.k", term_early_at="b0")),
        rc.Matcher("b1.a.head6").chain(rc.restrict("a.v", term_early_at="b0")),
        rc.Matcher("b1.a.head6").chain(rc.restrict("a.q", term_early_at="b0")),
        rc.Matcher("b1.a.head6").chain(rc.restrict("a.k", term_early_at="b0")),

    ])
    res["not-baseline"] = make_corr([non_ind_heads])
    res["not-baseline-full"] = make_corr(
        [non_ind_heads | rc.IterativeMatcher(rc.restrict("b0", term_early_at="b1.a"))]
    )

    # EMBEDDING-VALUE
    ev = ind_v.chain("b0.a")
    res["ev"] = make_corr([ev])
    res["not-ev"] = make_corr([ind_v.chain(embeds)])

    # EMBEDDING-QUERY
    eq = ind_q.chain("b0.a")
    res["eq"] = make_corr([eq])
    res["not-eq"] = make_corr([ind_q.chain(embeds)])

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
        [rc.Matcher("outside_input_toks_int")],
        options={"split_paths_by_position": [
            (5, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
        ]}
    )

    res["positional-eq"] = make_corr(
        [rc.Matcher("outside_input_toks_int")],
        options={"split_paths_by_position": [
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
        ]}
    )

    res["positional-pth-k"] = make_corr(
        [
            (
                rc.IterativeMatcher("outside_input_toks_int") |
                ind_heads.chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
            ),
        ],
        options={"split_paths_by_position": [
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 1, 1),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 1, 1),
        ]}
    )

    # positional eq + ev + pth-k
    res["positional-all-naive"] = make_corr(
        [
            (
                rc.IterativeMatcher("outside_input_toks_int") |
                ind_heads.chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
            ),
        ],
        options={"split_paths_by_position": [
            (5, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 1, 1),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 1, 1),
        ]}
    )

    # positional eq + pth-k
    res["positional-eq-and-pth-k"] = make_corr(
        [
            (
                rc.IterativeMatcher("outside_input_toks_int") |
                ind_heads.chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
            )
        ],
        options={"split_paths_by_position": [
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 1, 1),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 1, 1),
        ]}
    )

    # Same as positional-eq but with longer induction context
    res["positional-eq-with-multi-tok-ind"] = make_corr(
        [
            (
                rc.IterativeMatcher("outside_input_toks_int")
            )
        ],
        options={"split_paths_by_position": [
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
        ]}
    )

    # Same as positional-pth-k but with longer induction context
    res["positional-pth-k-with-multi-tok-ind"] = make_corr(
        [
            (
                rc.IterativeMatcher("outside_input_toks_int") |
                ind_heads.chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
            )
        ],
        options={"split_paths_by_position": [
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
        ]}
    )

    # Same as positional-eq-and-pth-k but with longer induction context
    res["positional-eq-and-pth-k-with-multi-tok-ind"] = make_corr(
        [
            (
                rc.IterativeMatcher("outside_input_toks_int") |
                ind_heads.chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
            )
        ],
        options={"split_paths_by_position": [
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
        ]}
    )

    # Same as above but with positional-ek for 1.5
    res["positional-eq-and-pth-k-with-multi-tok-ind-with-1.5-positional-ek"] = make_corr(
        [
            (
                rc.IterativeMatcher("outside_input_toks_int") |
                rc.IterativeMatcher("b1.a.head6").chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
            )
        ],
        options={"split_paths_by_position": [
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 4),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
        ]}
    )

    # Same as positional-eq-and-pth-k-with-multi-tok-ind but with positional-ev as well
    res["positional-all-naive-with-multi-tok-ind"] = make_corr(
        [
            (
                rc.IterativeMatcher("outside_input_toks_int") |
                ind_heads.chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
            )
        ],
        options={"split_paths_by_position": [
            (5, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
        ]}
    )

    # Same as above but only for 1.5
    res["positional-all-naive-with-multi-tok-ind-1.5"] = make_corr(
       [
           (
            rc.IterativeMatcher("outside_input_toks_int") |
            rc.IterativeMatcher("b1.a.head5").chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
           )
       ],
        options={"split_paths_by_position": [
            (5, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
        ]}
    )

    # Same as above but only for 1.6
    res["positional-all-naive-with-multi-tok-ind-1.6"] = make_corr(
       [
           (
            rc.IterativeMatcher("outside_input_toks_int") |
            rc.IterativeMatcher("b1.a.head6").chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
           )
       ],
        options={"split_paths_by_position": [
            (6, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
        ]}
    )

    # Same as multi-tok-ind-1.5 but with current tok included for keys
    res["positional-all-naive-with-multi-tok-ind-1.5-with-positional-ek"] = make_corr(
        [ind_heads.chain(rc.Matcher("outside_input_toks_int"))],
        options={"split_paths_by_position": [
            (5, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 4),
        ]}
    )

    # Same as above but for 1.6
    res["positional-all-naive-with-multi-tok-ind-1.6-with-positional-ek"] = make_corr(
        [ind_heads.chain(rc.Matcher("outside_input_toks_int"))],
        options={"split_paths_by_position": [
            (6, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 4),
        ]}
    )

    res["positional-ek"] = make_corr(
        [ind_heads.chain(rc.Matcher("outside_input_toks_int"))],
        options={"split_paths_by_position": [
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
        ]}
    )

    res["positional-ek-1.5"] = make_corr(
        [ind_heads.chain(rc.Matcher("outside_input_toks_int"))],
        options={"split_paths_by_position": [
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
        ]}
    )

    # Positional EV + Positional EQ + Positional multi-tok Q + Positional multi-tok PTH-K + Positional EK for 1.5
    # Similar to positional-all-naive-with-multi-tok-ind, but with positional EK for 1.5
    res["positional-all-naive-with-multi-tok-ind-with-1.5-positional-ek"] = make_corr(
        [
            (
                ind_heads.chain(rc.Matcher("outside_input_toks_int")) |
                rc.IterativeMatcher("b1.a.head6").chain(rc.restrict("a.k", term_early_at="b0.a")).chain(rc.restrict("idxed_embeds", term_early_at="b0.a"))
            ),
         ],
        options={"split_paths_by_position": [
            (5, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (6, "v", [0, 1, 2, 3, 4, 5, 6, 7], 0, 1),
            (5, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (6, "q", [0, 1, 2, 3, 4, 5, 6, 7], 2, 3),
            (5, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 4),
            (6, "k", [0, 1, 2, 3, 4, 5, 6, 7], 3, 3),
        ]}
    )

    # Similar to above, but with more specificity about the heads
    # For both 1.5 and 1.6, limit values to current positions through heads 0.12356 and skip connection
    # For both 1.5 and 1.6, limit queries to most recent three positions through 0.06,
    # and current position through 0.1235 and skip connection
    # For 1.5, limit keys to the three previous positions through 0.06, and the current
    # position through 0.12356 and skip connection
    # For 1.6, limit keys to the three previous positions through 0.06
    res["positional-all-naive-with-multi-tok-ind-with-1.5-positional-ek-fine-grained"] = make_corr(
        [
            (
                ind_heads.chain(rc.Matcher("outside_input_toks_int")) |
                ind_heads.chain("a.v").chain(rc.Regex("b0.a.head[047]")) |
                ind_heads.chain("a.q").chain(rc.Regex("b0.a.head[47]")) |
                rc.Matcher("b1.a.head5").chain("a.k").chain(rc.Regex("b0.a.head[47]")) |
                rc.Matcher("b1.a.head6").chain("a.k").chain(rc.Regex("b0.a.head[123457]")) |
                rc.Matcher("b1.a.head6").chain(rc.restrict("a.k", term_early_at="b0.a").chain(rc.restrict("idxed_embeds", term_early_at="b0.a")))
            ),
         ],
        options={"split_paths_by_position": [
            (5, "v", [1, 2, 3, 5, 6], 0, 1),
            (6, "v", [1, 2, 3, 5, 6], 0, 1),
            (5, "q", [0, 6], 2, 3),
            (5, "q", [1, 2, 3, 5], 0, 1),
            (6, "q", [0, 6], 2, 3),
            (6, "q", [1, 2, 3, 5], 0, 1),
            (5, "k", [0], 3, 3),
            (5, "k", [6], 3, 4),
            (5, "k", [1, 2, 3, 5], 0, 1),
            (6, "k", [0, 6], 3, 3),
        ]}
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
    parser.add_argument("--save", action="store_true", dest="save")
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

    save_name = args.exp_name
    if args.idx is not None:
        experiments = make_experiments(partial(make_corr, sampler=FixedSampler(args.idx)))
        save_name = f"{args.exp_name}_saa_{args.idx}"

    run_experiment(experiments, args.exp_name, args.samples, save_name)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
