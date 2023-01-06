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

    # EMBEDDING-QUERY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({1})
        .chain("b0.a"),
    )
    res["eq"] = corr

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
