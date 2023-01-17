# %%
import argparse
from typing import Optional
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
    children: list[rc.IterativeMatcher] = [], options: Optional[dict[str, str]] = None, sampler=ExactSampler()
):
    """Make a correspondence graph using a specific sampler."""
    corr = Correspondence()
    i_root = InterpNode(sampler, name="logits", other_inputs_sampler=sampler)
    corr.add(i_root, corr_root_matcher)
    for i, child in enumerate(children):
        tmp = i_root.make_descendant(UncondSampler(), name=f"{i}")
        corr.add(tmp, child)
    return (corr, options, children)


def make_make_corr(sampler):
    """Make a correspondence maker."""
    return partial(make_corr, sampler=sampler)


def m(head: int):
    return rc.IterativeMatcher(f"b0.a.head{head}")


# EXPERIMENTS
def make_experiments(make_corr) -> dict[str, tuple[Correspondence, dict[str, Optional[str]]]]:
    res = {}

    def make_corr_i(args: list[rc.IterativeMatcher]):
        """Make corr but split heads individually"""
        return make_corr(args, options={"split_heads": "b0-all"})

    def make_corr_a(args: list[rc.IterativeMatcher]):
        """Make corr but split all heads individually"""
        return make_corr(args, options={"split_heads": "all"})

    # shortcut matchers for useful parts of the graph
    embeds = rc.restrict("idxed_embeds", term_early_at="b0.a")
    a1_head = rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True))
    v = a1_head.children_matcher({3})
    q = a1_head.children_matcher({1})
    k = a1_head.children_matcher({2})

    # UNSCRUBBED
    res["unscrubbed"] = make_corr()

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
            res[f"swap-{l}{h}-{l2}{h2}"] = make_corr(options={"split_heads": "all", "swap": ((l, h), (l2, h2))})

    # BASELINE
    res["baseline"] = make_corr([rc.IterativeMatcher("a1.ind")])
    res["not-baseline"] = make_corr([rc.IterativeMatcher("a1.not_ind")])
    res["not-baseline-full"] = make_corr(
        [rc.IterativeMatcher("a1.not_ind") | rc.IterativeMatcher(rc.restrict("b0", term_early_at="b1.a"))]
    )
    for i in range(8):
        res[f"0.{i}"] = make_corr_a([rc.IterativeMatcher(f"b0.a.head{i}")])
        res[f"1-0.{i}"] = make_corr_a([rc.IterativeMatcher("b1.a").chain(f"b0.a.head{i}")])
        res[f"resid-0.{i}"] = make_corr_a([rc.IterativeMatcher(rc.restrict(f"b0.a.head{i}", term_early_at="b1.a"))])
        res[f"1.{i}"] = make_corr_a([rc.IterativeMatcher(f"b1.a.head{i}")])

    res[f"resid-0"] = make_corr_a([rc.IterativeMatcher(rc.restrict("b0", term_early_at="b1.a"))])
    res[f"resid-0-indiv"] = make_corr_a(
        [rc.IterativeMatcher(rc.restrict(m(i), term_early_at="b1.a")) for i in range(8)]
    )
    res[f"resid-0-prev"] = make_corr_a([rc.IterativeMatcher(rc.restrict(m(0) | m(6), term_early_at="b1.a"))])
    res[f"resid-0-begin"] = make_corr_a([rc.IterativeMatcher(rc.restrict(m(4) | m(7), term_early_at="b1.a"))])
    res[f"resid-0-diag"] = make_corr_a(
        [rc.IterativeMatcher(rc.restrict(m(1) | m(2) | m(3) | m(5), term_early_at="b1.a"))]
    )

    # EMBEDDING-VALUE
    ev = v.chain("b0.a")

    res["v"] = make_corr([v])
    res["ev"] = make_corr([ev])
    res["not-ev"] = make_corr([v.chain(embeds)])

    # scrub each head individually
    for i in range(8):
        res[f"a0-v-{i}"] = make_corr_i([v.chain(embeds | m(i))])
        res[f"a0-v-only-{i}"] = make_corr_i([v.chain(m(i))])

    # scrub all heads ind.
    res[f"a0-v-indep"] = make_corr_i([v.chain(m(i)) for i in range(8)] + [v.chain(embeds)])
    res[f"a0-v-indep-only"] = make_corr_i([v.chain(m(i)) for i in range(8)])

    # scrub subsets of heads
    res[f"a0-v-0,6"] = make_corr_i([v.chain(embeds | m(0) | m(6))])
    res[f"a0-v-only0,6"] = make_corr_i([v.chain(m(0) | m(6))])
    res[f"a0-v-only016"] = make_corr_i([v.chain(m(0) | m(1) | m(6))])
    res[f"a0-v-not0,6"] = make_corr_i([v.chain(embeds | m(1) | m(2) | m(3) | m(4) | m(5) | m(7))])
    res[f"a0-v-onlynot0,6"] = make_corr_i([v.chain(m(1) | m(2) | m(3) | m(4) | m(5) | m(7))])
    res[f"a0-v-all-bad-but-2"] = make_corr_i([v.chain(embeds | m(1) | m(3) | m(4) | m(5) | m(7))])
    res[f"a0-v-all-bad-but-12"] = make_corr_i([v.chain(embeds | m(3) | m(4) | m(5) | m(7))])

    # EMBEDDING-QUERY
    eq = q.chain("b0.a")
    res["q"] = make_corr([q])
    res["eq"] = make_corr([eq])
    res["not-eq"] = make_corr([q.chain(embeds)])

    # scrub each head individually
    for i in range(8):
        res[f"a0-q-{i}"] = make_corr_i([q.chain(embeds | m(i))])
        res[f"a0-q-only-{i}"] = make_corr_i([q.chain(m(i))])

    # scrub all heads ind.
    res[f"a0-q-indep"] = make_corr_i([q.chain(m(i)) for i in range(8)] + [q.chain(embeds)])
    res[f"a0-q-indep-only"] = make_corr_i([q.chain(m(i)) for i in range(8)])
    res[f"a0-q-indep-125"] = make_corr_i([q.chain(m(i)) for i in [2, 5, 1]])
    res[f"a0-q-indep-1235"] = make_corr_i([q.chain(m(i)) for i in [2, 5, 1, 3]])
    res[f"a0-q-indep-12356"] = make_corr_i([q.chain(m(i)) for i in [2, 5, 1, 3, 6]])

    # scrub subsets of heads
    res[f"a0-q-only-56"] = make_corr_i([q.chain(m(5) | m(6))])

    # EMBEDDING-KEY
    res["k"] = make_corr([k])
    res["ek"] = make_corr([k.chain("b0.a")])
    res["not-ek"] = make_corr([k.chain(embeds)])

    # PREVIOUS-TOKEN-HEAD KEY
    pth_k = k.chain(embeds | rc.Regex(r"\.*not_prev\.*"))
    res["pth-k"] = make_corr([pth_k])
    res["pth-k-full"] = make_corr(
        [
            pth_k
            | k.chain("a.attn_probs * a.not_prev_tok_mask")
            | k.chain("a.attn_probs * a.prev_tok_mask").chain("a.attn_probs")
        ],
        options={"split_pth_ov_by_pt_or_not": True},
    )
    res["not-pth-k"] = make_corr([k.chain(embeds | rc.Regex(r"\.*yes_prev\.*"))])
    res["pth-k-fine"] = make_corr([k.chain(rc.Regex(r"\.*not_prev\.*"))])
    res["not-pth-k-emb"] = make_corr([k.chain(embeds | rc.Regex(r"\.*yes_prev\.*"))])

    # include 0.6 also
    res["not-pth-k-emb-06"] = make_corr_i([k.chain(embeds | m(0) | m(6))])
    res["not-pth-k-06"] = make_corr_i([k.chain(m(0) | m(6))])
    res["not-pth-k-6"] = make_corr_i([k.chain(m(6))])
    res["not-pth-k-7"] = make_corr_i([k.chain(m(7))])
    res["not-pth-unimp"] = make_corr_i([k.chain(m(1) | m(2) | m(3) | m(4) | m(5) | m(7))])

    # PTH-QUERY
    res["pth-q"] = make_corr([q.chain(embeds | rc.Regex(r"\.*not_prev\.*"))])

    # ALL (3 og scrubbing)
    res["all"] = make_corr([ev, eq, pth_k])

    return res


def run(experiments, exp_name, samples, save_name, verbose):
    options = experiments[exp_name][1] or {}
    model, good_induction_candidate, tokenizer, toks_int_values = construct_circuit(**options)

    res, c_res, lc_res, scrubbed_circuit, inps = run_experiment(
        experiments,
        exp_name,
        model,
        toks_int_values,
        good_induction_candidate,
        tokenizer,
        verbose=verbose,
        samples=samples,
        save_name=save_name,
    )
    torch.cuda.empty_cache()
    return res, c_res, lc_res, scrubbed_circuit, inps, tokenizer


# %%
def main():
    experiments = make_experiments(make_make_corr(ExactSampler()))

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run causal scrubbing experiments on induction heads in the 2L attn-only model"
    )
    parser.add_argument(
        "--exp", action="store", dest="exp_name", type=str, choices=list(experiments.keys()), default="unscrubbed"
    )
    parser.add_argument("--samples", action="store", dest="samples", type=int, default=10000)
    parser.add_argument("--verbose", action="store", dest="verbose", type=int, default=0)
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

    save_name = ""
    if args.save:
        save_name = args.exp_name
    if args.idx is not None:
        experiments = make_experiments(make_make_corr(FixedSampler(args.idx)))
        if args.save:
            save_name += f"_saa_{args.idx}"

    run(experiments, args.exp_name, args.samples, save_name, args.verbose)


if __name__ == "__main__":
    main()
