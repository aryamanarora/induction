# %%
from main import *
import argparse

# EXPERIMENTS
def make_experiments():
    res = {}

    # UNSCRUBBED
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    corr.add(i_root, corr_root_matcher)
    res["unscrubbed"] = corr

    # BASELINE
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    a1_ind = i_root.make_descendant(UncondSampler(), name="a1.ind")
    corr.add(i_root, corr_root_matcher)
    corr.add(a1_ind, rc.IterativeMatcher("a1.ind"))
    res["baseline"] = corr

    # EMBEDDING-VALUE
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain("b0.a"),
    )
    res["ev"] = corr

    # NOT EMBEDDING-VALUE
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain(rc.restrict("idxed_embeds", end_depth=3)),
    )
    res["not-ev"] = corr

    # VALUE
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v, rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({3})
    )
    res["v"] = corr

    # EMBEDDING-QUERY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    q = i_root.make_descendant(UncondSampler(), name="a1.ind.q_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        q,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({1})
        .chain("b0.a"),
    )
    res["eq"] = corr

    # NOT EMBEDDING-QUERY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    q = i_root.make_descendant(UncondSampler(), name="a1.ind.q_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        q,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({1})
        .chain("b0")
        .chain(rc.restrict("idxed_embeds", end_depth=2)),
    )
    res["not-eq"] = corr

    # QUERY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    q = i_root.make_descendant(UncondSampler(), name="a1.ind.q")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        q, rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({1})
    )
    res["q"] = corr

    # KEY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        k, rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({2})
    )
    res["k"] = corr

    # PTH-KEY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        k,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({2})
        .chain(
            rc.restrict("idxed_embeds", term_early_at="b0.a")  # direct embeds not going through a0
            | rc.Regex(r"\.*not_prev\.*")  # not-previous-token heads
        ),
    )
    res["pth-k"] = corr

    # PTH-QUERY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        k,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({1})
        .chain(
            rc.restrict("idxed_embeds", term_early_at="b0.a")  # direct embeds not going through a0
            | rc.Regex(r"\.*not_prev\.*")  # not-previous-token heads
        ),
    )
    res["pth-q"] = corr

    # ALL (3 og scrubbing)
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    q = i_root.make_descendant(UncondSampler(), name="a1.ind.q_input")
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain("b0.a"),
    )
    corr.add(
        q,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({1})
        .chain("b0.a"),
    )
    corr.add(
        k,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({2})
        .chain(
            rc.restrict("idxed_embeds", term_early_at="b0.a")  # direct embeds not going through a0
            | rc.Regex(r"\.*not_prev\.*")  # not-previous-token heads
        ),
    )
    res["all"] = corr

    return res


def main():
    experiments = make_experiments()

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run causal scrubbing experiments on induction heads in the 2L attn-only model"
    )
    parser.add_argument(
        "--exp", action="store", dest="exp_name", type=str, choices=list(experiments.keys()), default="unscrubbed"
    )
    parser.add_argument("--runs", action="store", dest="runs", type=int, default=1)
    args = parser.parse_args()
    print(args)

    loss, good_induction_candidate, tokenizer, toks_int_values = construct_circuit()
    with_a1_ind_inputs = clean_model(loss)

    run_experiment(
        experiments,
        args.exp_name,
        with_a1_ind_inputs,
        toks_int_values,
        good_induction_candidate,
        tokenizer,
        runs=args.runs,
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# %%
