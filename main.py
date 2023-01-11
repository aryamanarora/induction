import pickle
from pprint import pprint
import rust_circuit as rc
import torch
from interp.circuit.causal_scrubbing.hypothesis import (Correspondence)
from interp.circuit.causal_scrubbing.experiment import Experiment, ExperimentEvalSettings
from interp.circuit.causal_scrubbing.dataset import Dataset

MAIN = __name__ == "__main__"
DEVICE = "cuda:0"


def get_induction_candidate_masks(
    inp_ixes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    inp_ixes is a 1d Tensor of ints representing the dataset indices we are interested in.
    Return two 2d Tensors of bools indicating, for each dataset row specified in inp_ixes, whether each token in that row is a good induction candidate (in the second tensor, we exclude first occurrences of each token in each row)
    """
    with open("data/mask_candidates.pkl", "rb") as f:
        candidates_mask = pickle.load(f)
    with open("data/mask_repeat_candidates.pkl", "rb") as f:
        repeat_candidates_mask = pickle.load(f)
    return candidates_mask[inp_ixes], repeat_candidates_mask[inp_ixes]


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
    ind_candidates_mask, ind_candidates_later_occur_mask = get_induction_candidate_masks(inp_ixes)

    if save_name:
        with open(f"data/{save_name}.pkl", "wb") as f:
            pickle.dump((res, ind_candidates_mask, ind_candidates_later_occur_mask), f)
        with open(f"data/inps_{save_name}.pkl", "wb") as f:
            pickle.dump(inps, f)
    return res, ind_candidates_mask, ind_candidates_later_occur_mask, scrubbed_circuit, inps


def get_inputs_from_model(model: rc.Circuit):
    data = model.get_unique("true_toks_int").get_unique("toks_int_var")
    return data.evaluate()


def run_experiment(
    exps, exp_name: str, model: rc.Circuit, toks, candidates, tokenizer, samples=10000, save_results=False, verbose=0
):
    save_name = f"{exp_name}" if save_results else ""
    res, ind_candidates_mask, ind_candidates_later_occur_mask, scrubbed_circuit, inps = run_hypothesis(
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
    c_res = res[ind_candidates_mask]
    print(f"{c_res.mean().item():>10.3f}{c_res.var().item():>10.3f}{c_res.shape[0]:>10}")
    # print(f"unnormed {(res * ind_candidates_mask).mean():.3f}")

    print("LATER CANDIDATES")
    lc_res = res[ind_candidates_later_occur_mask]
    print(f"{lc_res.mean().item():>10.3f}{lc_res.var().item():>10.3f}{lc_res.shape[0]:>10}")

    return res, c_res, lc_res, inps