from experiments import *
from interp import cui
from matplotlib import pyplot as plt


def main():
    idx = 4
    experiments = make_experiments(make_make_corr(FixedSampler(idx)))

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Compare causal scrubbing experiments on induction heads in the 2L attn-only model"
    )
    parser.add_argument(
        "--exp1", action="store", dest="exp_name", type=str, choices=list(experiments.keys()), default="unscrubbed"
    )
    parser.add_argument(
        "--exp2", action="store", dest="exp2_name", type=str, choices=list(experiments.keys()), default="baseline"
    )
    parser.add_argument("--samples", action="store", dest="samples", type=int, default=10000)
    parser.add_argument("--verbose", action="store", dest="verbose", type=int, default=0)
    args = parser.parse_args()
    print(args)

    res1, c_res1, lc_res1, inps1, tokenizer = run(
        experiments, args.exp_name, args.samples if args.exp_name != "unscrubbed" else idx + 1, False, args.verbose
    )
    res2, c_res2, lc_res2, inps2, _ = run(
        experiments, args.exp2_name, args.samples if args.exp2_name != "unscrubbed" else idx + 1, False, args.verbose
    )

    mean1 = res1.mean(dim=0)
    mean2 = res2.mean(dim=0)
    res = (mean1 - mean2).cpu().reshape(15, 1, -1)
    labs = [x.replace("\n", "\\n").replace(" ", "_") for x in tokenizer.batch_decode(inps1.reshape(-1))]
    print(res)

    fig, ax = plt.subplots(15, 1)
    m = res.abs().max().item()
    print(m)
    for i in range(15):
        ax[i].imshow(res[i], vmin=-m, vmax=m, cmap="RdBu")
        ax[i].set_xticks(list(range(res.shape[2])))
        ax[i].set_yticks([])
        print(labs[i * 20 : i * 20 + 21])
        ax[i].set_xticklabels(labs[i * 20 + 1 : i * 20 + 21], rotation=45)
    print("plotting")
    plt.gcf().set_size_inches(10, 30)
    plt.savefig("diff.png")


if __name__ == "__main__":
    main()
