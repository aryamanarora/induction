# %%
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from adjustText import adjust_text

from experiments import make_experiments
from main import run_experiment
from typing import Optional
from tqdm import tqdm


def run_pairs(
    experiments: dict[str, tuple],
    exp: str,
    layer1: int,
    layer2: int,
    samples: int = 500,
    repeat: bool = True,
    l: Optional[list] = None,
):
    """Run experiments which involve two heads, either in composition or swapping or something more complex.
    Inputs:
    - experiments: Dictionary of experiments.
    - exp: Experiment name. E.g. "swap"-l1h1-l2h2
    - layer1: The layer from which scrubbing/swapping happens.
    - layer2: The layer to which scrubbing/swapping happens.
    - repeat: Whether to look at whole matrix or only the upper triangle of comparisons (e.g. 1-2 but not 2-1).
    - l: list of sub-scrubs to run
    """
    comp = {}

    # labels/experiment names for layer2 (include scrub of embed/all if possible)
    if l is None:
        l = [f"-{layer2}.{i}" for i in range(8)]
        if f"{exp}-{layer1}.0-emb" in experiments:
            l.append("-emb")
        if f"{exp}-{layer1}.0" in experiments:
            l.append("")
    print(l)

    pca = PCA(n_components=2)
    loss_vectors = [run_experiment(experiments, "unscrubbed", samples)[0].reshape(-1).cpu().numpy()]

    # run experiments
    exps: list[tuple[str, int]] = [("unscrubbed", 0)]
    for h in range(8):
        for i in tqdm(range((h + 1) if not repeat else 0, len(l))):
            name = f"{layer1}.{h}{l[i]}"
            exps.append((name, h))
            res, _, _ = run_experiment(experiments, exp + "-" + name, samples)
            torch.cuda.empty_cache()
            loss = res.mean().item()
            loss_vectors.append(res.reshape(-1).cpu().numpy())
            comp[(h, i)] = (4.631 - loss) / (4.631 - 4.2)
            print(h, i, comp[(h, i)])

    # histogram
    fig, axs = plt.subplots(4, 2)
    for i in range(1, len(loss_vectors)):
        axs[(i - 1) // 2][(i - 1) % 2].hist(
            loss_vectors[i] - loss_vectors[0], label=exps[i][0], bins=100, range=(-7.5, 12.5)
        )
        axs[(i - 1) // 2][(i - 1) % 2].set_yscale("log")
    plt.show()
    plt.clf()

    # pca
    compressed = pca.fit_transform(np.array(loss_vectors))
    print("Compressed")
    texts = []
    for i, (x, y) in enumerate(compressed):
        plt.scatter([x], [y])
        texts.append(plt.annotate(exps[i][0], (x, y), alpha=0.5))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", alpha=0.5))
    plt.title(exp)
    plt.savefig(f"figs/{exp}-pca.png", dpi=400)
    plt.show()
    plt.clf()

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
    plt.imshow(g, vmin=g.min().item(), vmax=g.max().item(), cmap="RdBu")
    plt.title(exp)
    plt.ylabel(f"Layer {layer1}")
    plt.xlabel(f"Layer {layer2}")
    plt.yticks(list(range(8)), labels=[f"{layer1}.{x}" for x in range(8)])
    plt.xticks(list(range(len(l))), labels=[x[1:] for x in l])
    plt.show()


def main():
    experiments = make_experiments()
    # l = [f"-0.{i}" for i in range(8)] + ["", "-emb", "-0.06", "-0.1235", "-0.47", "-0"]
    l = [""]
    run_pairs(experiments, "all", 1, 1, l=l)


if __name__ == "__main__":
    main()

# %%
