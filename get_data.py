from experiments import make_make_corr, make_experiments, FixedSampler
from main import run_experiment
from tqdm import tqdm
import argparse


def run(exps: list[str], attns: bool, attn_scores: bool, ct: int):
    name = "saa"
    if attns:
        name = "attns"
    if attn_scores:
        name = "attn_scores"
    for i in tqdm(range(ct)):
        experiments = make_experiments(make_make_corr(FixedSampler(i)))
        for exp in exps:
            save_name = f"{exp}_{name}_{i}"
            run_experiment(experiments, exp, 1000, save_name, 0, get_attns=attns, get_attn_scores=attn_scores)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Save inference runs for visualisation on 2L attn only models")
    parser.add_argument("--exp", action="store", dest="exp_name", nargs="+", type=str, default=["unscrubbed"])
    parser.add_argument("--attns", action="store_true", dest="attns")
    parser.add_argument("--attn-scores", action="store_true", dest="attn_scores")
    parser.add_argument("--ct", action="store", dest="count", type=int, default=50)
    args = parser.parse_args()

    run(args.exp_name, args.attns, args.attn_scores, args.count)


if __name__ == "__main__":
    main()
