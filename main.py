import pytz
import pickle
import os.path
from pprint import pprint
from datetime import datetime

import torch

import utils
import rust_circuit as rc
from interp.circuit.causal_scrubbing.hypothesis import (Correspondence)
from interp.circuit.causal_scrubbing.experiment import Experiment, ExperimentEvalSettings
from interp.circuit.causal_scrubbing.dataset import Dataset

MAIN = __name__ == "__main__"
DEVICE = "cuda:0"
RESULTS_PATH = "results"
DATA_PATH = "data"


def get_induction_candidate_masks(
    inp_ixes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    inp_ixes is a 1d Tensor of ints representing the dataset indices we are interested in.
    Return two 2d Tensors of bools indicating, for each dataset row specified in inp_ixes, whether each token in that row is a good induction candidate (in the second tensor, we exclude first occurrences of each token in each row)
    """
    with open(os.path.join(DATA_PATH, "mask_candidates.pkl"), "rb") as f:
        candidates_mask = pickle.load(f)
    with open(os.path.join(DATA_PATH, "mask_repeats.pkl"), "rb") as f:
        repeats_mask = pickle.load(f)
    with open(os.path.join(DATA_PATH, "mask_repeat_candidates.pkl"), "rb") as f:
        repeat_candidates_mask = pickle.load(f)
    return candidates_mask[inp_ixes], repeats_mask[inp_ixes], repeat_candidates_mask[inp_ixes]


@torch.inference_mode()
def run_hypothesis(
    circuit: rc.Circuit,
    toks: rc.Array,
    correspondence: Correspondence,
    good_induction_candidates,
    samples=10000,
    tokenizer=None,
    verbose=0,
    seed: int = 42,
    save_name="",
):
    if verbose:
        print("Running hypothesis")
    ds = Dataset(arrs={"toks_int_var": toks})
    eval_settings = ExperimentEvalSettings(device_dtype=DEVICE, batch_size=100, run_on_all=True)

    exp = Experiment(circuit, ds, correspondence, num_examples=samples, random_seed=seed)
    scrubbed_circuit = exp.scrub()
    inps = get_inputs_from_model(scrubbed_circuit.circuit)
    inp_ixes = inps[:, -1]
    inps = inps[:, :-1]
    if verbose:
        scrubbed_circuit.print()
    res = scrubbed_circuit.evaluate(eval_settings)
    if verbose == 2:
        if tokenizer is not None:
            pprint(tokenizer.batch_decode(inps))
            binps = inps.clone()
            binps[get_induction_candidate_masks(inp_ixes)[0]] = inps[0][0]
            pprint(tokenizer.batch_decode(binps))

    if verbose:
        print("Building induction candidates masks")
    ind_candidates_mask, repeats_mask, ind_candidates_later_occur_mask = get_induction_candidate_masks(inp_ixes)

    if save_name:
        meta = {
            "save_name"       : save_name,
            "seed"            : seed,
            "samples"         : samples,
            "timestamp"       : datetime.now(pytz.timezone("America/Los_Angeles")),
        }
        with open(os.path.join(RESULTS_PATH, f"{save_name}.pkl"), "wb") as f:
            pickle.dump((res, inp_ixes, meta), f)
    return res, ind_candidates_mask, repeats_mask, ind_candidates_later_occur_mask, scrubbed_circuit, inps, inp_ixes


def get_inputs_from_model(model: rc.Circuit):
    data = model.get_unique("true_toks_int").get_unique("toks_int_var")
    return data.evaluate()


def run_experiment(
    exps, exp_name: str, model: rc.Circuit, toks, candidates, tokenizer, samples=10000, save_name="", verbose=0
):
    res, ind_candidates_mask, repeats_mask, ind_candidates_later_occur_mask, scrubbed_circuit, inps, inp_ixes = run_hypothesis(
        model,
        toks,
        exps[exp_name][0],
        candidates,
        samples=samples,
        tokenizer=tokenizer,
        verbose=verbose,
        seed=42,
        save_name=save_name,
    )

    print(exp_name.upper())
    print("OVERALL")
    print(f"{res.mean().item():>10.3f}{res.var().item():>10.3f}{res.shape[0] * res.shape[1]:>10}")

    print("CANDIDATES")
    # Apply mask to INPUT
    c_res = res[ind_candidates_mask[:, :-1]]
    print(f"{c_res.mean().item():>10.3f}{c_res.var().item():>10.3f}{c_res.shape[0]:>10}")
    # print(f"unnormed {(res * ind_candidates_mask).mean():.3f}")

    print("LATER CANDIDATES")
    # Apply mask to INPUT
    lc_res = res[ind_candidates_later_occur_mask[:, :-1]]
    print(f"{lc_res.mean().item():>10.3f}{lc_res.var().item():>10.3f}{lc_res.shape[0]:>10}")

    print("REPEATS")
    # Apply mask to OUTPUT
    r_res = res[repeats_mask[:, 1:]]
    print(f"{r_res.mean().item():>10.3f}{r_res.var().item():>10.3f}{r_res.shape[0]:>10}")

    print("UNCOMMON REPEATS")
    # Apply mask to OUTPUT
    untop_200_mask = utils.build_token_frequency_filter(200)[inp_ixes]
    ur_mask = repeats_mask.logical_and(untop_200_mask)
    ur_res = res[ur_mask[:, 1:]]
    print(f"{ur_res.mean().item():>10.3f}{ur_res.var().item():>10.3f}{ur_res.shape[0]:>10}")

    print("NON-ERB UNCOMMON REPEATS")
    with open(os.path.join(DATA_PATH, "mask_ends_of_repeated_bigrams.pkl"), "rb") as f:
        erb_mask = pickle.load(f)[inp_ixes]
    nerb_ur_mask = erb_mask.logical_not().logical_and(ur_mask)
    # Apply mask to OUTPUT
    neur_res = res[nerb_ur_mask[:, 1:]]
    print(f"{neur_res.mean().item():>10.3f}{neur_res.var().item():>10.3f}{neur_res.shape[0]:>10}")

    return res, c_res, lc_res, scrubbed_circuit, inps
