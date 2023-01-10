# %%
from main import *
import argparse
from typing import Optional


def make_corr(children: list[rc.IterativeMatcher] = [], options: Optional[dict[str, str]] = None):
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    corr.add(i_root, corr_root_matcher)
    for i, child in enumerate(children):
        tmp = i_root.make_descendant(UncondSampler(), name=f"{i}")
        corr.add(tmp, child)
    return (corr, options)


# EXPERIMENTS
def make_experiments() -> dict[str, tuple[Correspondence, dict[str, Optional[str]]]]:
    res = {}

    # UNSCRUBBED
    res["unscrubbed"] = make_corr()

    # BASELINE
    res["baseline"] = make_corr([rc.IterativeMatcher("a1.ind")])
    res["not-baseline"] = make_corr([rc.IterativeMatcher("a1.not_ind")])

    # EMBEDDING-VALUE
    v_matcher = (
        rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({3})
    )
    ev = v_matcher.chain("b0.a")

    res["v"] = make_corr([v_matcher])
    res["ev"] = make_corr([ev])
    res["not-ev"] = make_corr([v_matcher.chain(rc.restrict("idxed_embeds", end_depth=3))])

    # EMBEDDING-QUERY
    q_matcher = (
        rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({1})
    )
    eq = q_matcher.chain("b0.a")

    res["q"] = make_corr([q_matcher])
    res["eq"] = make_corr([eq])
    res["not-eq"] = make_corr([q_matcher.chain(rc.restrict("idxed_embeds", end_depth=3))])

    # EMBEDDING-KEY
    k_matcher = (
        rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({2})
    )

    res["k"] = make_corr([k_matcher])
    res["ek"] = make_corr([k_matcher.chain("b0.a")])
    res["not-ek"] = make_corr([k_matcher.chain(rc.restrict("idxed_embeds", end_depth=3))])

    # PREVIOUS-TOKEN-HEAD KEY
    pth_k = k_matcher.chain(rc.restrict("idxed_embeds", term_early_at="b0.a") | rc.Regex(r"\.*not_prev\.*"))
    res["pth-k"] = make_corr([pth_k])
    res["not-pth-k"] = make_corr(
        [k_matcher.chain(rc.restrict("idxed_embeds", term_early_at="b0.a") | rc.Regex(r"\.*yes_prev\.*"))]
    )
    res["pth-k-fine"] = make_corr([k_matcher.chain(rc.Regex(r"\.*not_prev\.*"))])
    res["not-pth-k-emb"] = make_corr(
        [k_matcher.chain(rc.restrict("idxed_embeds", term_early_at="b0.a") | rc.Regex(r"\.*yes_prev\.*"))]
    )

    # PTH-QUERY
    res["pth-q"] = make_corr(
        [q_matcher.chain(rc.restrict("idxed_embeds", term_early_at="b0.a") | rc.Regex(r"\.*not_prev\.*"))]
    )

    # ALL (3 og scrubbing)
    res["all"] = make_corr([ev, eq, pth_k])

    # a0.7
    res["a0.7"] = make_corr([rc.IterativeMatcher("b0.a.head7")], options={"split_heads": "b0-all"})
    res["a0.0"] = make_corr([rc.IterativeMatcher("b0.a.head0")], options={"split_heads": "b0-all"})

    return res


def a0_heads_to_v() -> dict[str, tuple[Correspondence, dict[str, Optional[str]]]]:
    """Check performance when scrubbing each head in layer 0 as input to a1_ind.k"""

    def make_corr_v(args: list[rc.IterativeMatcher]):
        """Make corr but split heads individually"""
        return make_corr(args, options={"split_heads": "b0-all"})

    def m(head: int):
        return rc.IterativeMatcher(f"b0.a.head{head}")

    exps = {}
    v_matcher = (
        rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({3})
    )
    embeds = rc.restrict("idxed_embeds", term_early_at="b0.a")

    # scrub each head individually
    for i in range(8):
        exps[f"a0-v-{i}"] = make_corr_v([v_matcher.chain(embeds | rc.IterativeMatcher(f"b0.a.head{i}"))])

    # scrub subsets of heads
    exps[f"a0-v-0,6"] = make_corr_v([v_matcher.chain(embeds | m(0) | m(6))])
    exps[f"a0-v-only0,6"] = make_corr_v([v_matcher.chain(m(0) | m(6))])
    exps[f"a0-v-only016"] = make_corr_v([v_matcher.chain(m(0) | m(1) | m(6))])
    exps[f"a0-v-not0,6"] = make_corr_v([v_matcher.chain(embeds | m(1) | m(2) | m(3) | m(4) | m(5) | m(7))])
    exps[f"a0-v-onlynot0,6"] = make_corr_v([v_matcher.chain(m(1) | m(2) | m(3) | m(4) | m(5) | m(7))])
    exps[f"a0-v-all-bad-but-2"] = make_corr_v([v_matcher.chain(embeds | m(1) | m(3) | m(4) | m(5) | m(7))])
    exps[f"a0-v-all-bad-but-12"] = make_corr_v([v_matcher.chain(embeds | m(3) | m(4) | m(5) | m(7))])

    return exps


def main():
    experiments = make_experiments()
    experiments.update(a0_heads_to_v())

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
    args = parser.parse_args()
    print(args)

    loss, good_induction_candidate, tokenizer, toks_int_values = construct_circuit()
    options = experiments[args.exp_name][1] or {}
    with_a1_ind_inputs = clean_model(loss, **options)

    run_experiment(
        experiments,
        args.exp_name,
        with_a1_ind_inputs,
        toks_int_values,
        good_induction_candidate,
        tokenizer,
        verbose=args.verbose,
        samples=args.samples,
        save_results=args.save,
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()