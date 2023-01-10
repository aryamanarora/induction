# %%
from main import *
import argparse
from typing import Optional

# EXPERIMENTS
def make_experiments() -> dict[str, tuple[Correspondence, dict[str, Optional[str]]]]:
    res = {}

    # UNSCRUBBED
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    corr.add(i_root, corr_root_matcher)
    res["unscrubbed"] = (corr, None)

    # BASELINE
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    a1_ind = i_root.make_descendant(UncondSampler(), name="a1.ind")
    corr.add(i_root, corr_root_matcher)
    corr.add(a1_ind, rc.IterativeMatcher("a1.ind"))
    res["baseline"] = (corr, None)

    # BASELINE
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    a1_ind = i_root.make_descendant(UncondSampler(), name="a1.ind")
    corr.add(i_root, corr_root_matcher)
    corr.add(a1_ind, rc.IterativeMatcher("a1.not_ind"))
    res["not-baseline"] = (corr, None)

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
    res["ev"] = (corr, None)

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
    res["not-ev"] = (corr, None)

    # VALUE
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v, rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({3})
    )
    res["v"] = (corr, None)

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
    res["eq"] = (corr, None)

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
    res["not-eq"] = (corr, None)

    # QUERY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    q = i_root.make_descendant(UncondSampler(), name="a1.ind.q")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        q, rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({1})
    )
    res["q"] = (corr, None)

    # EMBEDDING-KEY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        k,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({2})
        .chain("b0.a"),
    )
    res["ek"] = (corr, None)

    # NOT EMBEDDING-KEY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        k,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({2})
        .chain("b0")
        .chain(rc.restrict("idxed_embeds", end_depth=2)),
    )
    res["not-ek"] = (corr, None)

    # KEY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        k, rc.IterativeMatcher("a1.ind").chain(rc.restrict("a.head.on_inp", term_if_matches=True)).children_matcher({2})
    )
    res["k"] = (corr, None)

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
    res["pth-k"] = (corr, None)

    # PTH-KEY (FINE)
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        k,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({2})
        .chain(rc.Regex(r"\.*not_prev\.*")),  # not-previous-token heads
    )
    res["pth-k-fine"] = (corr, None)

    # NOT PTH-KEY
    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    k = i_root.make_descendant(UncondSampler(), name="a1.ind.k_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        k,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({2})
        .chain(rc.Regex(r"\.*yes_prev\.*")),  # previous-token heads
    )
    res["not-pth-k"] = (corr, None)

    # NOT PTH-KEY + EMBED
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
            | rc.Regex(r"\.*yes_prev\.*")  # previous-token heads
        ),
    )
    res["not-pth-k-emb"] = (corr, None)

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
    res["pth-q"] = (corr, None)

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
    res["all"] = (corr, None)

    return res


def a0_heads_to_v() -> dict[str, tuple[Correspondence, dict[str, Optional[str]]]]:
    """Check performance when scrubbing each head in layer 0 as input to a1_ind.k"""
    exps = {}
    for i in range(8):
        corr = Correspondence()
        i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
        v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
        corr.add(i_root, corr_root_matcher)
        corr.add(
            v,
            rc.IterativeMatcher("a1.ind")
            .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
            .children_matcher({3})
            .chain(rc.restrict("idxed_embeds", term_early_at="b0.a") | rc.IterativeMatcher(f"b0.a.head{i}")),
        )
        exps[f"a0-v-{i}"] = (corr, {"split_heads": "b0-all"})

    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain(
            rc.restrict("idxed_embeds", term_early_at="b0.a")
            | rc.IterativeMatcher(f"b0.a.head0")
            | rc.IterativeMatcher(f"b0.a.head6")
        ),
    )
    exps[f"a0-v-0,6"] = (corr, {"split_heads": "b0-all"})

    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain(rc.IterativeMatcher(f"b0.a.head0") | rc.IterativeMatcher(f"b0.a.head6")),
    )
    exps[f"a0-v-only0,6"] = (corr, {"split_heads": "b0-all"})

    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain(
            rc.restrict("idxed_embeds", term_early_at="b0.a")
            | rc.IterativeMatcher(f"b0.a.head1")
            | rc.IterativeMatcher(f"b0.a.head2")
            | rc.IterativeMatcher(f"b0.a.head3")
            | rc.IterativeMatcher(f"b0.a.head4")
            | rc.IterativeMatcher(f"b0.a.head5")
            | rc.IterativeMatcher(f"b0.a.head7")
        ),
    )
    exps[f"a0-v-not0,6"] = (corr, {"split_heads": "b0-all"})

    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain(
            rc.IterativeMatcher(f"b0.a.head1")
            | rc.IterativeMatcher(f"b0.a.head2")
            | rc.IterativeMatcher(f"b0.a.head3")
            | rc.IterativeMatcher(f"b0.a.head4")
            | rc.IterativeMatcher(f"b0.a.head5")
            | rc.IterativeMatcher(f"b0.a.head7")
        ),
    )
    exps[f"a0-v-onlynot0,6"] = (corr, {"split_heads": "b0-all"})

    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain(
            rc.restrict("idxed_embeds", term_early_at="b0.a")
            | rc.IterativeMatcher(f"b0.a.head1")
            | rc.IterativeMatcher(f"b0.a.head3")
            | rc.IterativeMatcher(f"b0.a.head4")
            | rc.IterativeMatcher(f"b0.a.head5")
            | rc.IterativeMatcher(f"b0.a.head7")
        ),
    )
    exps[f"a0-v-all-bad-but-2"] = (corr, {"split_heads": "b0-all"})

    corr = Correspondence()
    i_root = InterpNode(ExactSampler(), name="logits", other_inputs_sampler=ExactSampler())
    v = i_root.make_descendant(UncondSampler(), name="a1.ind.v_input")
    corr.add(i_root, corr_root_matcher)
    corr.add(
        v,
        rc.IterativeMatcher("a1.ind")
        .chain(rc.restrict("a.head.on_inp", term_if_matches=True))
        .children_matcher({3})
        .chain(
            rc.restrict("idxed_embeds", term_early_at="b0.a")
            | rc.IterativeMatcher(f"b0.a.head3")
            | rc.IterativeMatcher(f"b0.a.head4")
            | rc.IterativeMatcher(f"b0.a.head5")
            | rc.IterativeMatcher(f"b0.a.head7")
        ),
    )
    exps[f"a0-v-all-bad-but-12"] = (corr, {"split_heads": "b0-all"})

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
    parser.add_argument("--runs", action="store", dest="runs", type=int, default=1)
    parser.add_argument("--verbose", action="store", dest="verbose", type=int, default=0)
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
        runs=args.runs,
        verbose=args.verbose,
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# %%
