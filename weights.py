# %%
import rust_circuit as rc
import torch
from interp.circuit.interop_rust.module_library import load_model_id
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
from tqdm import tqdm

plt.rcParams["figure.dpi"] = 50

DEVICE = "cuda:0"

model_id = "attention_only_2"
(loaded, tokenizer, extra_args) = load_model_id(model_id)
loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE)) for (s, c) in loaded.items()}

orig_circuit = loaded["t.bind_w"]
tok_embeds = loaded["t.w.tok_embeds"].value
pos_embeds = loaded["t.w.pos_embeds"].value
all_toks = [x.replace("\n", "\\n").replace(" ", "_") for x in tokenizer.batch_decode(torch.arange(tok_embeds.shape[0]))]

# attention weights
wk = loaded["t.bind_w"].get_unique("a0.w.k_arr").value
wq = loaded["t.bind_w"].get_unique("a0.w.q_arr").value
wv = loaded["t.bind_w"].get_unique("a0.w.v_arr").value
wo = loaded["t.bind_w"].get_unique("a0.w.o_arr").value

# layernormed embeddings
ln_bias = loaded["t.bind_w"].get_unique(f"a0.ln.w.bias_arr").value
ln_scale = loaded["t.bind_w"].get_unique(f"a0.ln.w.scale_arr").value
lnormed_embeds = torch.nn.functional.layer_norm(tok_embeds, (tok_embeds.shape[1],), ln_scale, ln_bias)
tok_norms = torch.norm(tok_embeds, dim=1)
lnormed_norms = torch.norm(lnormed_embeds, dim=1)


def print_toks(tensor: torch.Tensor, vals: torch.Tensor):
    """Print a token and some value of it from a vector"""
    for i in tensor:
        t = all_toks[i]
        print(f"{t:<20}{vals[i].item():<20.3f}")


def embed_plot(embed, comp=2, title="sus"):
    """Plot PCA of some embeddings"""
    pca = PCA(n_components=comp)
    compressed = pca.fit_transform(embed.cpu())
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    if comp == 2:
        plt.scatter(
            [x for (x, y) in compressed],
            [y for (x, y) in compressed],
            cmap="RdBu",
            c=list(range(len(compressed))),
            alpha=0.5,
        )
    plt.title(f"{title} ({sum(pca.explained_variance_ratio_):.3f})")
    plt.show()
    plt.clf()

    for i in range(comp):
        plt.plot(list(range(len(compressed))), [x[i] for x in compressed])
    plt.show()


def top_norms(embed: torch.Tensor):
    """Print the top 10 and bottom 10 vectors by norm"""
    norms = torch.norm(embed, dim=1, p=2)
    norm_sorted_order = norms.sort().indices
    print(norm_sorted_order)
    norm_sorted = embed[norm_sorted_order]
    print_toks(norm_sorted_order[:10], norms)
    print_toks(norm_sorted_order[-10:], norms)


def top_matrix(matrix: torch.Tensor):
    s = matrix.shape
    matrix = matrix.reshape(-1)
    order = matrix.sort().indices[-10:]
    for x in order:
        i, j = x // s[1], x % s[1]
        print(f"{all_toks[i]:<20}{all_toks[j]:<20}{matrix[x].item():<20.3f}")


def plot_layernorm():
    """Plot pre- and post-layernorm embedding stuff"""
    ln_bias = loaded["t.bind_w"].get_unique(f"a0.ln.w.bias_arr").value
    ln_scale = loaded["t.bind_w"].get_unique(f"a0.ln.w.scale_arr").value
    lnormed_embeds = torch.nn.functional.layer_norm(tok_embeds, (tok_embeds.shape[1],))
    plt.plot(lnormed_embeds[50258].cpu(), label="[BEGIN]")
    plt.plot(ln_scale.cpu(), label="ln.scale")
    plt.xlabel("dim")
    plt.legend()
    plt.show()
    plt.clf()

    lnormed_embeds = torch.nn.functional.layer_norm(tok_embeds, (tok_embeds.shape[1],), ln_scale, ln_bias)
    print(top_norms(lnormed_embeds))
    plt.plot(lnormed_embeds[50258].cpu(), label="[BEGIN]")
    plt.plot(ln_scale.cpu(), label="ln.scale")
    plt.xlabel("dim")
    plt.legend()
    plt.show()
    plt.clf()

    check = [50258, 286, 198]

    tok_norms = torch.norm(tok_embeds, dim=1).cpu()
    lnormed_norms = torch.norm(lnormed_embeds, dim=1).cpu()
    plt.scatter(tok_norms, lnormed_norms, alpha=0.5)
    plt.xlabel("pre-lnorm")
    plt.ylabel("lnormed")
    texts = []
    for i in check:
        texts.append(plt.text(tok_norms[i], lnormed_norms[i], all_toks[i]))
    # adjust_text(texts, arrowprops=dict(arrowstyle="->"), text_from_points=False, text_from_text=False)
    plt.show()

    fig, axs = plt.subplots(4, 2)
    for i in range(8):
        a0_v = torch.einsum("ve,ae -> va", lnormed_embeds.cpu(), wv[i].cpu())
        a0_v = torch.einsum("va,ea -> ve", a0_v, wo[i].cpu())
        a0_v_norms = torch.norm(a0_v, dim=1).cpu()
        axs[i // 2][i % 2].set_xlabel("lnormed")
        axs[i // 2][i % 2].set_ylabel(f"a0.{i}")
        texts = []
        for j in check:
            texts.append(axs[i // 2][i % 2].text(lnormed_norms[j], a0_v_norms[j], all_toks[j]))
        # adjust_text(texts, arrowprops=dict(arrowstyle="->"), text_from_points=False, text_from_text=False)
        axs[i // 2][i % 2].scatter(lnormed_norms.cpu(), a0_v_norms.cpu(), alpha=0.5)
    plt.show()

    return lnormed_embeds


# embed_plot(tok_embeds, title="tok_embeds")
# lnormed_embeds = plot_layernorm(loaded, tok_embeds).cpu()
# embed_plot(lnormed_embeds, title="lnormed_embeds")


def pairwise_cosine_sim():
    """Plot pairwise cosine similarities between OV matrices of layer 0 heads"""
    normalised_lnormed = torch.div(lnormed_embeds.t().cpu(), lnormed_norms.cpu()).t().cpu()
    normalised_embeds = torch.div(tok_embeds.t().cpu(), tok_norms.cpu()).t().cpu()

    g = torch.zeros((9, 9))

    for i in range(8):
        # get normed OV projected vectors for this head
        a = torch.einsum("ve,ae -> va", lnormed_embeds, wv[i].cpu())
        a = torch.einsum("va,ea -> ve", a, wo[i].cpu())
        a = torch.div(a.t(), torch.norm(a, dim=1)).t()

        # compare with all other heads
        for j in range(i + 1, 8):
            b = torch.einsum("ve,ae -> va", lnormed_embeds, wv[j].cpu())
            b = torch.einsum("va,ea -> ve", b, wo[j].cpu())
            b = torch.div(b.t(), torch.norm(b, dim=1)).t()
            sim = torch.einsum("ve,ve -> v", a, b)
            g[i][j] = sim.mean().item()
            g[j][i] = g[i][j]
            print("=" * 10, "\n", i, j, f"{g[i][j]:.3f}")
            top_norms(sim.reshape(50259, 1))

        # compare w embeds
        sim = torch.einsum("ve,ve -> v", a, normalised_embeds.cpu())
        g[i][8] = sim.mean().item()
        g[8][i] = g[i][8]
        print("=" * 10, "\n", i, 8, f"{g[i][8]:.3f}")
        top_norms(sim.reshape(50259, 1))

    plt.imshow(g, vmin=-1, vmax=1, cmap="RdBu")
    plt.show()


# %%
