# %%
import torch
import torch.nn.functional as F

import rust_circuit as rc
import matplotlib.pyplot as plt

from main import run_experiment, Dataset, ExperimentEvalSettings, DEVICE, Experiment, get_inputs_from_model
from model import construct_circuit
from experiments import make_experiments, make_make_corr, ExactSampler, FixedSampler
import pickle
import utils


def compare():
    """Compare the loss differences between scrubbing 1.5, 1.6, and 1.0."""
    data_10 = []
    data_15 = []
    data_16 = []
    inps = utils.load_inputs()
    print(inps)
    unscrubbed = []

    for i in range(50):
        with open(f"results/1.0_saa_{i}.pkl", "rb") as f:
            data = pickle.load(f)
            data_10.extend(data[0].mean(0).tolist())
        with open(f"results/1.5_saa_{i}.pkl", "rb") as f:
            data = pickle.load(f)
            data_15.extend(data[0].mean(0).tolist())
        with open(f"results/1.6_saa_{i}.pkl", "rb") as f:
            data = pickle.load(f)
            data_16.extend(data[0].mean(0).tolist())
        with open(f"results/unscrubbed_saa_{i}.pkl", "rb") as f:
            data = pickle.load(f)
            unscrubbed.extend(data[0].mean(0).tolist())

    data_10 = torch.tensor(data_10).cpu()
    data_15 = torch.tensor(data_15).cpu()
    data_16 = torch.tensor(data_16).cpu()
    unscrubbed = torch.tensor(unscrubbed)
    diff_10 = data_10 - unscrubbed
    diff_15 = data_15 - unscrubbed
    diff_16 = data_16 - unscrubbed

    d1 = diff_15.reshape(50, 300)
    d2 = diff_16.reshape(50, 300)
    for i in range(50):
        for j in range(300):
            if d1[i][j] < d2[i][j] - 2 and j + 2 < 300:
                print(d1[i][j] - d2[i][j], d1[i][j + 1] - d2[i][j + 1], d1[i][j + 2] - d2[i][j + 2])

    plt.scatter(diff_15, diff_16, alpha=0.1)
    plt.xlabel("1.5")
    plt.ylabel("1.6")
    plt.show()
    plt.clf()

    plt.scatter(diff_15, diff_10, alpha=0.1)
    plt.xlabel("1.5")
    plt.ylabel("1.0")
    plt.show()
    plt.clf()

    plt.scatter(diff_16, diff_10, alpha=0.1)
    plt.xlabel("1.6")
    plt.ylabel("1.0")
    plt.show()
    plt.clf()


def plot(layer, head, graph=False):
    """Plot similarity between dot products of outputs of different parts of the model."""
    model, good_induction_candidate, tokenizer, toks_int_values = construct_circuit(split_heads="all")
    model = rc.substitute_all_modules(model)
    ds = Dataset(arrs={"toks_int_var": toks_int_values})
    eval_settings = ExperimentEvalSettings(device_dtype=DEVICE, batch_size=1, run_on_all=True)

    corr = make_make_corr(ExactSampler())()[0]
    exp = Experiment(model, ds, corr, num_examples=1, random_seed=42)
    scrubbed_circuit = exp.scrub()
    res = scrubbed_circuit.evaluate(eval_settings).cpu()

    sampler = eval_settings.get_sampler(1, scrubbed_circuit.group)
    emb = sampler.sample(scrubbed_circuit.circuit.get_unique("idxed_embeds")).evaluate().cpu()
    ln = sampler.sample(scrubbed_circuit.circuit.get_unique("a0.norm")).evaluate().cpu()
    b0 = (
        sampler.sample(
            scrubbed_circuit.circuit.get_unique(
                rc.IterativeMatcher(f"b{layer}.a.head{head}").chain(rc.restrict("a.v", end_depth=4))
            )
        )
        .evaluate()
        .cpu()
    )

    if graph:
        emb_norm = torch.einsum("bve,bve->bv", emb, emb).cpu().reshape(-1)
        plt.scatter(list(range(emb_norm.shape[0])), emb_norm)
        plt.title("emb norm")
        plt.show()
        plt.clf()

        ln_norm = torch.einsum("bve,bve->bv", ln, ln).cpu().reshape(-1)
        plt.scatter(list(range(ln_norm.shape[0])), ln_norm)
        plt.title("ln norm")
        plt.show()
        plt.clf()

        b0_norm = torch.einsum("bve,bve->bv", b0, b0).cpu().reshape(-1)
        plt.scatter(list(range(b0_norm.shape[0])), b0_norm)
        plt.title("a0 norm")
        plt.show()
        plt.clf()

    sim_emb = torch.einsum("bse,bte->bst", emb, emb)
    sim_b0 = torch.einsum("bse,bte->bst", b0, b0)
    m = sim_emb.abs().max().item() / 10
    # plt.imshow(sim_emb, vmin=-m, vmax=m, cmap='RdBu')
    plt.scatter(sim_emb.reshape(-1), sim_b0.reshape(-1), alpha=0.1)
    plt.title(f"a{layer}.{head}")
    plt.xlabel("emb dot product")
    plt.ylabel("head dot product")
    plt.show()
    plt.clf()


def main():
    compare()


if __name__ == "__main__":
    main()

# %%
