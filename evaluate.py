import argparse
import pickle
import torch
import glob

from masks import get_all_masks

def load_exp_results(exp_name):
    '''
    Load exp_name results from either results/{exp_name}.pkl
    or results/positional_scrubs/{exp_name}_*.pkl
    '''
    try:
        return consolidate_results(exp_name, "positional_scrubs")
    except FileNotFoundError:
        with open(f"results/{exp_name}.pkl", "rb") as f:
            res, ixes, _ = pickle.load(f)
            return res, ixes
    

def filter_and_align(res, ixes, common_ixes):
    '''
    Keep only unique res and ix for which the ix is in common_ixes
    Return them sorted by ixes
    '''
    keep_ixes = [ix.item() in common_ixes for ix in ixes]
    res, ixes = res[keep_ixes], ixes[keep_ixes]
    _, sorted_ixes = torch.sort(ixes)
    res, ixes = res[sorted_ixes], ixes[sorted_ixes]
    is_unique = torch.cat([torch.tensor([True], device='cuda'), ixes[1:] != ixes[:-1]]) # Wow! Thanks copilot
    return res[is_unique], ixes[is_unique]


def consolidate_results(exp_name, results_subdir):
    '''
    Consolidate all exp_name results in results_subdir into single res and ixes vars
    as if they were generated jointly via the --save flag in experiments.py
    Assume they were pickled as res, _, inp_ixes, _
    '''
    exp_fns = glob.glob(f"results/{results_subdir}/{exp_name}_*.pkl")
    if not exp_fns:
        raise FileNotFoundError
    all_res = []
    all_inp_ixes = []
    for fn in exp_fns:
        with open(fn, "rb") as f:
            res, _, inp_ixes, _ = pickle.load(f)
            all_res.append(res)
            all_inp_ixes.append(inp_ixes)
    return torch.cat(all_res), torch.cat(all_inp_ixes)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an experimental result against a baseline and a target, either for behavior recoveries (point-to-point comparisons) or loss recoveries (expectation)."
    )
    parser.add_argument(
        "--exp",
        action="store",
        dest="exp_eval",
        type=str,
        default="unscrubbed",
        help="Experiment to evaluate",
    )
    parser.add_argument(
        "-b", "--baseline",
        action="store",
        dest="exp_baseline",
        type=str,
        default="baseline",
        help="Baseline experiment to compare --exp against for the recovery calculation"
    )
    parser.add_argument(
        "-t", "--target",
        action="store",
        dest="exp_target",
        type=str,
        default="unscrubbed",
        help="Target experiment to compare --exp against"
    )
    parser.add_argument(
        "-e",
        action="store_true",
        dest="expected_loss",
        help="Whether to calculate loss recoveries (set this flag) or behavior recoveries (don't set this flag)"
    )
    parser.add_argument(
        "-s",
        action="store_true",
        dest="squared_distance",
        help="Whether to calculate squared distance (set this flag) or absolute distance (don't set this flag) for behavior recoveries. Assumes -e flag was NOT set."
    )
    
    args = parser.parse_args()
    assert not(args.expected_loss and args.squared_distance)
    
    exp_eval_res, exp_eval_ixes = load_exp_results(args.exp_eval)
    exp_baseline_res, exp_baseline_ixes = load_exp_results(args.exp_baseline)
    exp_target_res, exp_target_ixes = load_exp_results(args.exp_target)

    # Get common ixes
    exp_eval_ixes_set = set(exp_eval_ixes.cpu().tolist())
    exp_baseline_ixes_set = set(exp_baseline_ixes.cpu().tolist())
    exp_target_ixes_set = set(exp_target_ixes.cpu().tolist())
    common_ixes = exp_eval_ixes_set.intersection(exp_baseline_ixes_set).intersection(exp_target_ixes_set)
    
    exp_eval_res, exp_eval_ixes = filter_and_align(exp_eval_res, exp_eval_ixes, common_ixes)
    exp_baseline_res, exp_baseline_ixes = filter_and_align(exp_baseline_res, exp_baseline_ixes, common_ixes)
    exp_target_res, exp_target_ixes = filter_and_align(exp_target_res, exp_target_ixes, common_ixes)
    
    assert torch.all(exp_eval_ixes == exp_baseline_ixes)
    assert torch.all(exp_eval_ixes == exp_target_ixes)
    
    if not args.expected_loss:
        dist_func = torch.square if args.squared_distance else torch.abs
        exp_baseline_res = dist_func(exp_target_res - exp_baseline_res)
        exp_eval_res = dist_func(exp_target_res - exp_eval_res)

    all_masks = get_all_masks(exp_eval_ixes)
    evals = [
        ("OVERALL", torch.ones_like(exp_baseline_res, dtype=torch.bool)),
        ("CANDIDATES", all_masks["induction_candidates"]),
        ("LATER CANDIDATES", all_masks["repeat_candidates"]),
        ("REPEATS", all_masks["repeats"]),
        ("UR", all_masks["uncommon_repeats"]),
        ("C UR", all_masks["c_uncommon_repeats"]),
        ("NC UR", all_masks["nc_uncommon_repeats"]),
        ("MISLEADING INDUCTION", all_masks["misleading_induction"]),
    ]
    print(args.exp_eval, args.exp_baseline, "Recovery", "Var", "Shape", sep="\t")
    data_row = []
    for eval_name, mask in evals:
        print(eval_name)
        masked_exp_baseline_res = exp_baseline_res[mask]
        masked_exp_eval_res = exp_eval_res[mask]
        baseline_mean = masked_exp_baseline_res.mean().item()
        baseline_var = masked_exp_baseline_res.var().item()
        eval_mean = masked_exp_eval_res.mean().item()
        eval_var = masked_exp_eval_res.var().item()
        shape = masked_exp_baseline_res.shape[0]
        if args.expected_loss:
            target_mean = exp_target_res[mask].mean().item()
            recovery = (baseline_mean - eval_mean) / (baseline_mean - target_mean)
        else:
            recovery = 1 - eval_mean / baseline_mean
        print(f"{eval_mean:.3f}", f"{baseline_mean:.3f}", f"{(recovery):.3f}", f"{eval_var:>10.3f}", f"{shape}", sep="\t")
        data_row += [str(eval_mean), str(recovery)]

    print(f"Num examples: {exp_eval_ixes.shape}")

    print(",".join(data_row))

if __name__ == "__main__":
    main()