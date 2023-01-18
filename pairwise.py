# %%
import torch
import matplotlib.pyplot as plt

from experiments import make_experiments
from main import run_experiment


def run_pairs(experiments: dict[str, tuple], exp: str, layer1: int, layer2: int, samples: int = 1000, repeat: bool = True):
    """Run pairwise comparisons between heads of one layer and another (or same layer).
    Inputs:
    - experiments: Dictionary of experiments.
    - exp: Experiment name.
    - layer1: The layer from which scrubbing/swapping happens.
    - layer2: The layer to which scrubbing/swapping happens.
    - repeat: Whether to look at whole matrix or only the upper triangle of comparisons (e.g. 1-2 but not 2-1).
    """
    comp = {}

    # labels/experiment names for layer2 (include scrub of embed/all if possible)
    l = [f"-{layer2}.{i}" for i in range(8)]
    if f"{exp}-{layer1}.0-emb" in experiments:
        l.append("-emb")
    if f"{exp}-{layer1}.0" in experiments:
        l.append("")

    # run experiments
    for h in range(8):
        for i in range((h + 1) if not repeat else 0, len(l)):
            res, _, _ = run_experiment(experiments, f"{exp}-{layer1}.{h}{l[i]}", 1000, verbose=0)
            torch.cuda.empty_cache()
            loss = res.mean().item()
            comp[(h, i)] = (4.631 - loss) / (4.631 - 4.2)
            print(h, i, comp[(h, i)])

    # fill in matrix for plot
    g = torch.zeros((8, len(l)))
    for x in comp:
        g[x[0]][x[1]] = comp[x]
        if not repeat:
            g[x[1]][x[0]] = comp[x]
    if not repeat:
        for i in range(8):
            g[i][i] = 1

    # plot
    plt.imshow(g, vmin=0.5, vmax=1, cmap="RdBu")
    plt.title(exp)
    plt.ylabel(f"Layer {layer1}")
    plt.xlabel(f"Layer {layer2}")
    plt.xticks(list(range(len(l))), labels=[x[1:] for x in l])
    plt.show()


def main():
    experiments = make_experiments()
    run_pairs(experiments, "k", 1, 0)


if __name__ == "__main__":
    main()

# %%
