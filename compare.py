from interp import cui
from matplotlib import pyplot as plt
import argparse
import torch
from tqdm import tqdm

from experiments import make_experiments, make_make_corr, FixedSampler
from main import run_experiment
from utils import load_tokenizer


def compare(exp1, exp2, num_samples, idx, verbose, plot=True):
    experiments = make_experiments(make_make_corr(FixedSampler(idx)))

    assert exp1 in list(experiments.keys())
    assert exp2 in list(experiments.keys())

    tokenizer = load_tokenizer()
    res1, _, inps1, _ = run_experiment(
        experiments, exp1, num_samples if exp1 != "unscrubbed" else idx + 1, "", verbose
    )
    res2, _, inps2, _ = run_experiment(experiments, exp2, num_samples if exp2 != "unscrubbed" else idx + 1, "", verbose)

    mean1 = res1.mean(dim=0)
    mean2 = res2.mean(dim=0)
    res = (mean1 - mean2).cpu()
    labs = [x.replace("\n", "\\n").replace(" ", "_") for x in tokenizer.batch_decode(inps1[0].reshape(-1))]

    if plot:
        fig, ax = plt.subplots(15, 1)
        m = res.abs().max().item()
        print(m)
        res = res.reshape(15, 1, -1)
        for i in range(15):
            ax[i].imshow(res[i], vmin=-m, vmax=m, cmap="RdBu")
            ax[i].set_xticks(list(range(res.shape[2])))
            ax[i].set_yticks([])
            ax[i].set_xticklabels(labs[i * 20 + 1 : i * 20 + 21], rotation=45)
        plt.gcf().set_size_inches(10, 25)
        plt.savefig(f"figs/{exp1}_{exp2}_{idx}.png")

    return res, res1, res2, inps1, inps2, tokenizer


def plot_convergence(exp1, exp2, idx, verbose):
    """Plot convergence of mean loss differences per token vs. scrubbing sample size."""
    ress = torch.zeros(1, 300)
    for i in tqdm(range(1, 500, 25)):
        res, _, _, _, _, _ = compare(exp1, exp2, i, idx, verbose, False)
        res = res.reshape(1, -1)
        ress = torch.cat((ress, res), 0)

    for i in range(300):
        plt.plot(list(range(1, 1000, 50)), ress[1:, i])
    plt.xlabel("Samples")
    plt.ylabel("Loss difference")
    plt.title(f"{exp1} vs. {exp2} on {idx}")
    plt.savefig(f"figs/convergence.png")


def plot_distribution(exp1, exp2, idx, verbose):
    """Plot distribution of loss differences between fixed inputed and varying scrubbed inputs."""
    res, res1, res2, inps1, inps2, tokenizer = compare(exp1, exp2, 10000, idx, verbose, False)
    l = (res1.mean(dim=0) - res2).mean(dim=1).cpu()
    _, indices = torch.sort(l)
    for i in indices[:1]:
        print(i, l[i])
        # print(tokenizer.batch_decode(inps1[i].reshape(-1)))
        print(tokenizer.batch_decode(inps2[i].reshape(-1)))
    # print(l)
    plt.hist(l.tolist(), bins=100)
    plt.ylabel("Samples")
    plt.xlabel("Loss difference")
    plt.title(f"{exp1} vs. {exp2} on {idx}")
    plt.savefig(f"figs/distribution.png")


def main():

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Compare causal scrubbing experiments on induction heads in the 2L attn-only model"
    )
    parser.add_argument("--exp1", action="store", dest="exp_name", type=str, default="unscrubbed")
    parser.add_argument("--exp2", action="store", dest="exp2_name", type=str, default="baseline")
    parser.add_argument("--type", action="store", dest="type", type=str, default="compare")
    parser.add_argument("--samples", action="store", dest="samples", type=int, default=1000)
    parser.add_argument("--idx", action="store", dest="idx", type=int, default=0)
    parser.add_argument("--verbose", action="store", dest="verbose", type=int, default=0)

    args = parser.parse_args()
    print(args)

    if args.type == "compare":
        compare(args.exp_name, args.exp2_name, args.samples, args.idx, args.verbose)
    elif args.type == "distrib":
        plot_distribution(args.exp_name, args.exp2_name, args.idx, args.verbose)
    elif args.type == "con":
        plot_convergence(args.exp_name, args.exp2_name, args.idx, args.verbose)


if __name__ == "__main__":
    main()
