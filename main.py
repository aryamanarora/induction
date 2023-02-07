import pytz
import pickle
import os.path
from pprint import pprint
from datetime import datetime
import csv

import torch

from masks import get_all_masks
import rust_circuit as rc
from interp.circuit.causal_scrubbing.hypothesis import Correspondence
from interp.circuit.causal_scrubbing.experiment import Experiment, ExperimentEvalSettings
from interp.circuit.causal_scrubbing.dataset import Dataset

from model import construct_circuit

MAIN = __name__ == "__main__"
DEVICE = "cuda:0"
RESULTS_PATH = "results"
DATA_PATH = "data"
SEED = 42


def get_inputs_from_model(model: rc.Circuit):
    data = model.get_unique("true_toks_int").get_unique("toks_int_var")
    return data.evaluate()


@torch.inference_mode()
def run_experiment(exps, exp_name, samples=10000, save_name="", verbose=0, get_attns=False, get_attn_scores=False):
    """Run a scrubbing experiments.
    Inputs:
    - exps: A dictionary of experiment names mapped to correspondences and options.
    - exp_name: Experiment name.
    - samples: Number of samples to run from dataset.
    - save_name: File to save info in.
    - verbose: How much to print during experiment."""
    corr, options, _ = exps[exp_name]
    options = options or {}
    model, _, tokenizer, toks = construct_circuit(**options)

    if verbose > 0:
        print("Running hypothesis")
    ds = Dataset(arrs={"toks_int_var": toks})
    eval_settings = ExperimentEvalSettings(device_dtype=DEVICE, batch_size=100, run_on_all=True)

    exp = Experiment(model, ds, corr, num_examples=samples, random_seed=SEED)
    scrubbed_circuit = exp.scrub()

    inps = get_inputs_from_model(scrubbed_circuit.circuit)
    inp_ixes = inps[:, -1]
    all_masks = get_all_masks(inp_ixes)
    inps = inps[:, :-1]

    if verbose > 0:
        scrubbed_circuit.print()

    # either get attention probs or losses
    if get_attns or get_attn_scores:
        # first sample (for computability), then substitute to subtrees are computable
        sampler = eval_settings.get_sampler(len(scrubbed_circuit.ref_ds), scrubbed_circuit.group)
        circ = rc.substitute_all_modules(sampler.sample(scrubbed_circuit.circuit))
        check = "a.attn_probs_sample" if get_attns else "a.attn_scores_sample"

        # collect attn scores for each head
        res = []
        for l in range(2):
            for h in range(8):
                c = list(
                    circ.get(rc.IterativeMatcher(f"b{l}.a.head{h}_sample").chain(rc.restrict(check, end_depth=4)))
                    if l == 1
                    else circ.get(
                        rc.IterativeMatcher("b1.a.head0_sample")
                        .chain(f"b{l}.a.head{h}_sample")
                        .chain(rc.restrict(check, end_depth=4))
                    )
                )[0]
                attn = c.evaluate().cpu()

                # reshape to be batchable by cat
                if len(attn.shape) == 3:
                    attn = attn.reshape(attn.shape[0], 1, attn.shape[1], attn.shape[2])
                res.append(attn)

        # batch together head attns
        res = torch.cat(tuple(res), dim=1).mean(dim=0).unsqueeze(dim=0)
    else:
        res = scrubbed_circuit.evaluate(eval_settings)

    timestamp = datetime.now(pytz.timezone("America/Los_Angeles"))
    if save_name:
        meta = {
            "save_name": save_name,
            "seed": SEED,
            "samples": samples,
            "timestamp": timestamp,
            "attns": get_attns,
        }
        with open(os.path.join(RESULTS_PATH, f"{save_name}.pkl"), "wb") as f:
            pickle.dump((res, inp_ixes, meta), f)
    if verbose > 0:
        print(exp_name.upper())

    if verbose == 2:
        if tokenizer is not None:
            pprint(tokenizer.batch_decode(inps))
            binps = inps.clone()
            binps[all_masks["induction_candidates"]] = inps[0][0]
            pprint(tokenizer.batch_decode(binps))

    if verbose > 0:
        print("Building induction candidates masks")

    evals_dict = {"exp name": exp_name}
    if not get_attns and not get_attn_scores:
        # calculated masked losses
        eval_scores = []
        evals = [
            ("OVERALL", torch.ones_like(res, dtype=torch.bool)),
            ("CANDIDATES", all_masks["induction_candidates"]),
            ("LATER CANDIDATES", all_masks["repeat_candidates"]),
            ("REPEATS", all_masks["repeats"]),
            ("UNCOMMON REPEATS", all_masks["uncommon_repeats"]),
            ("NERB UR", all_masks["nerb_uncommon_repeats"]),
            ("MISLEADING INDUCTION", all_masks["misleading_induction"]),
            ("CANDIDATE ERB", all_masks["candidate_erb"]),
            ("NFERB UR", all_masks["nferb_uncommon_repeats"]),
        ]
        for eval_name, mask in evals:
            if verbose > -1:
                print(eval_name)
            masked_res = res[mask]
            mean = masked_res.mean().item()
            var = masked_res.var().item()
            shape = masked_res.shape[0]
            eval_scores.extend([mean, var, shape])
            evals_dict[eval_name] = (mean, var, shape)
            if verbose > -1:
                print(f"{mean:>10.3f}{var:>10.3f}{shape:>10}")

        # log the losses
        with open("logs.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([exp_name, SEED, samples, timestamp] + eval_scores)

    return res, scrubbed_circuit, inps, evals_dict
