from experiments import make_experiments, make_corr, FixedSampler
from main import run_experiment
from tqdm import tqdm
from functools import partial
import argparse
import csv

EXPS = ["unscrubbed"]

def run(exps: list[str], attns: bool, attn_scores: bool, ct: int, evals: bool):
    if evals:
        with open("out.csv", "w", newline="") as csvfile:
            fieldnames = ["exp name", "OVERALL", "LATER CANDIDATES", "NERB UR", "CANDIDATE ERB", "NFERB UR", "UNCOMMON REPEATS", "REPEATS", "MISLEADING INDUCTION", "CANDIDATES"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            experiments = make_experiments()
            for exp in tqdm(exps):
                _, _, _, evals_dict = run_experiment(experiments, exp)
                evals_dict = {k: v if k == "exp name" else v[0] for k, v in evals_dict.items()}
                writer.writerow(evals_dict)
        return

    name = "saa"
    if attns:
        name = "attns"
    if attn_scores:
        name = "attn_scores"
    for i in tqdm(range(ct)):
        experiments = make_experiments(partial(make_corr, sampler=FixedSampler(i)))
        for exp in exps:
            save_name = f"{exp}_{name}_{i}"
            run_experiment(experiments, exp, 1000, save_name, 0, get_attns=attns, get_attn_scores=attn_scores)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Save inference runs for visualisation on 2L attn only models")
    parser.add_argument("--exp", action="store", dest="exp_name", nargs="+", type=str, default=EXPS)
    parser.add_argument("--attns", action="store_true", dest="attns")
    parser.add_argument("--attn-scores", action="store_true", dest="attn_scores")
    parser.add_argument("--ct", action="store", dest="count", type=int, default=50)
    parser.add_argument("--evals", action="store_true", dest="evals")
    args = parser.parse_args()

    run(args.exp_name, args.attns, args.attn_scores, args.count, args.evals)


if __name__ == "__main__":
    main()
