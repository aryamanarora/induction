import pytz
import pickle
import os.path
from datetime import datetime

import torch

from tqdm import tqdm

import rust_circuit as rc
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
def run_experiment(exps, exp_name, samples=10240, save_name=""):
    """Run a scrubbing experiments.
    Inputs:
    - exps: A dictionary of experiment names mapped to correspondences and options.
    - exp_name: Experiment name.
    - samples: Number of samples to run from dataset.
    - save_name: File to save info in."""
    corr, options, _ = exps[exp_name]
    options = options or {}

    model, _, tokenizer, toks = construct_circuit(**options, save_name=exp_name)
    ds = Dataset(arrs={"toks_int_var": toks})

    experiment_size = 128
    seed = SEED
    eval_settings = ExperimentEvalSettings(
        device_dtype=DEVICE,
        batch_size=32,
        run_on_all=True,
        optim_settings=rc.OptimizationSettings(scheduling_naive=True)
    )
    num_iters = samples // experiment_size + (1 if samples % experiment_size else 0)
    for i in tqdm(range(num_iters)):
        try:
            with open(f"results/positional_scrubs/{save_name or exp_name}_{i}.pkl", "rb") as f:
                print("Skipping", i)
                seed += 1
                continue
        except FileNotFoundError:
            pass
        exp = Experiment(model, ds, corr, num_examples=experiment_size, random_seed=seed)
        scrubbed_circuit = exp.scrub()

        inps = get_inputs_from_model(scrubbed_circuit.circuit)
        inp_ixes = inps[:, -1]
        res = scrubbed_circuit.evaluate(eval_settings)
        with open(f"results/positional_scrubs/{save_name or exp_name}_{i}.pkl", "wb") as f:
            pickle.dump((res, inp_ixes, {"seed": seed}), f)
        seed += 1
        torch.cuda.empty_cache()