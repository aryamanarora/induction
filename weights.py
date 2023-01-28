# %%
import rust_circuit as rc
import torch
from interp.circuit.interop_rust.module_library import load_model_id
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from umap import UMAP
from sklearn.manifold import TSNE
import plotly.express as px

plt.rcParams["figure.dpi"] = 400

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
pos_norms = torch.norm(pos_embeds, dim=1)
lnormed_norms = torch.norm(lnormed_embeds, dim=1)

normalised_lnormed = torch.div(lnormed_embeds.t(), lnormed_norms).t()
normalised_embeds = torch.div(tok_embeds.t(), tok_norms).t()
normalised_pos = torch.div(pos_embeds.t(), pos_norms).t()

categories: list[str] = []


def categorise(n):
    if all_toks[n] in ["[BEGIN]", "[END]", "\n", "<|endoftext|>"]:
        return all_toks[n]
    t: str = all_toks[n].strip("_").lower()
    if len(t) == 0:
        return "_"
    if t.isdigit() and len(t) == 4:
        return "year"
    if t.isdigit():
        return "number"
    elif t in ["i", "he", "you", "she", "they", "we", "my", "her", "his", "me", "them", "him", "us"]:
        return "pronoun"
    elif t.endswith("ly"):
        return "ly-adverb"
    elif t.endswith("est"):
        return "superlative"
    elif t.endswith("ing"):
        return "gerund"
    elif t in [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]:
        return "month"
    elif t.endswith("day"):
        return "day"
    elif all_toks[n].isupper():
        return "uppercase"
    elif all_toks[n][0].isupper():
        return "capital"
    elif t[0] in ",.():/\\!@#$%^&*[}]{|":
        return "punctuation"
    elif t in ["to", "as", "with", "of", "in", "on", "at", "for"]:
        return "prep"
    return "other"


for i in range(len(all_toks)):
    categories.append(categorise(i))


def print_toks(tensor: torch.Tensor, vals: torch.Tensor):
    """Print a token and some value of it from a vector"""
    for i in tensor:
        t = all_toks[i]
        print(f"{t:<20}{vals[i].item():<20.3f}")


def embed_plot(embed, comp=2, title="sus", xy=False, uma=False):
    """Plot PCA of some embeddings"""
    if uma:
        pca = UMAP()
    else:
        pca = PCA(n_components=comp)
    compressed = pca.fit_transform(embed.cpu())
    if not uma:
        print(pca.explained_variance_ratio_)
        print(sum(pca.explained_variance_ratio_))
    if comp == 2:
        fig = px.scatter(
            x=[x for (x, y) in compressed],
            y=[y for (x, y) in compressed],
            opacity=0.5,
            color=categories,
            hover_name=all_toks,
            title=title,
        )
        fig.show()

    if xy:
        for i in range(comp):
            plt.plot(list(range(len(compressed))), [x[i] for x in compressed])
        plt.show()


def embed_plot_many(*embeds, labels=None, title="sus", pos=False, save=False):
    umap = UMAP(n_components=2)
    compressed = umap.fit_transform(torch.cat(embeds, dim=0).cpu())
    if labels is None:
        labels = list(range(len(embeds)))
    c = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    color = []
    x = [x for (x, y) in compressed]
    y = [y for (x, y) in compressed]

    for i, l in enumerate(labels):
        color.extend([(i, str(l)) for _ in range(embeds[i].shape[0])])

    if save:
        s = 0
        for i in range(len(embeds)):
            plt.scatter(
                x[s : s + len(embeds[i])],
                y[s : s + len(embeds[i])],
                color=c[i],
                label=labels[i],
                alpha=0.3,
            )
            s += len(embeds[i])
        plt.title(title)
        plt.legend()
        plt.savefig(f"saved_figs/kq/{title}.png")
        plt.show()
    else:
        fig = px.scatter(
            x=[x for (x, y) in compressed],
            y=[y for (x, y) in compressed],
            opacity=0.5,
            color=color,
            hover_name=(all_toks if not pos else list(range(len(embeds[0])))) * len(embeds),
            title=title,
        )
        fig.show()


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


def transform_vocab(layer=0, head=0, type="v", normalise=True, pos=False) -> torch.Tensor:
    idx = layer * 8 + head
    a: torch.Tensor
    if type == "v":
        a = torch.einsum("ve,ae -> va", pos_embeds if pos else lnormed_embeds, wv[idx])
        a = torch.einsum("va,ea -> ve", a, wo[idx])
    elif type == "k":
        a = torch.einsum("ve,ae -> va", pos_embeds if pos else lnormed_embeds, wk[idx])
    elif type == "q":
        a = torch.einsum("ve,ae -> va", pos_embeds if pos else lnormed_embeds, wq[idx])
    if normalise:
        a = torch.div(a.t(), torch.norm(a, dim=1)).t()
    return a


def pairwise_cosine_sim(type="v", verbose=False):
    """Plot pairwise cosine similarities between OV matrices of layer 0 heads"""

    g = torch.zeros((9, 9))

    for i in range(8):
        # get normed OV projected vectors for this head
        a = transform_vocab(0, i, type)
        embed_plot(a, title=f"0.{i} OV")

        # compare with all other heads
        for j in range(i + 1, 8):
            b = transform_vocab(0, j, type)
            sim = torch.einsum("ve,ve -> v", a, b)
            g[i][j] = g[j][i] = sim.mean().item()
            if verbose:
                print("=" * 10, "\n", i, j, f"{g[i][j]:.3f}")
                top_norms(sim.reshape(50259, 1))

        # compare w embeds
        if type == "v":
            sim = torch.einsum("ve,ve -> v", a, normalised_embeds)
            g[i][8] = g[8][i] = sim.mean().item()
            if verbose:
                print("=" * 10, "\n", i, 8, f"{g[i][8]:.3f}")
                top_norms(sim.reshape(50259, 1))

    plt.imshow(g, vmin=-1, vmax=1, cmap="RdBu")
    plt.show()


def kq(count=1000):
    plt.rcParams["figure.figsize"] = (5, 5)
    for p in [False]:
        for r in [False]:
            for head in range(8):
                k = transform_vocab(0, head, "k", normalise=False, pos=p)
                q = transform_vocab(0, head, "q", normalise=False, pos=r)
                vocab_k = k.shape[0]
                vocab_q = q.shape[0]
                print(head, p, r, vocab_k, vocab_q)

                vals, idxs = [], []
                for j in tqdm(range(0, vocab_k, 100)):
                    add = j * vocab_q
                    sim = torch.einsum("ke,qe -> kq", k[j : j + 100], q)
                    if p and r:
                        sim = sim.triu(diagonal=j)
                    _, i = torch.topk(sim.flatten().abs(), count)
                    v = sim.flatten()[i]
                    vals.append(v)
                    idxs.append((i + add).long())

                vals = torch.cat(vals, dim=0)
                idxs = torch.cat(idxs, dim=0)
                print(idxs)
                tops = torch.sort(vals.abs(), descending=True).indices

                name = ("p" if p else "t") + ("p" if r else "t")
                with open(f"saved_logs/qk/{name}-0.{head}.txt", "w") as f:
                    f.write(f"{'query':<20} {'key':<20} {'attn_score':<20}\n")
                    for j in range(count):
                        t = tops[j]
                        a = idxs[t] % vocab_k
                        b = idxs[t] // vocab_k
                        f.write(f"{a if p else all_toks[a]:<20} {b if r else all_toks[b]:<20} {vals[t]:<20.5f}\n")


def kq_graph():
    for head in range(8):
        k = transform_vocab(0, head, "k", normalise=False, pos=False)
        q = transform_vocab(0, head, "q", normalise=False, pos=False)
        k_pos = transform_vocab(0, head, "k", normalise=False, pos=True)
        q_pos = transform_vocab(0, head, "q", normalise=False, pos=True)
        embed_plot_many(k, q, k_pos, q_pos, labels=["tk", "tq", "pk", "pq"], title=f"0.{head}", save=True)


def autoencoder():
    plt.rcParams["figure.figsize"] = (40, 5)
    fig, axs = plt.subplots(1, 8)
    for head in range(8):
        ct_k = 0
        ct_q = 0
        avg_k = 0
        avg_q = 0
        all_k, all_q = [], []
        k = transform_vocab(0, head, "k", normalise=False)
        q = transform_vocab(0, head, "q", normalise=False)
        vocab_k = k.shape[0]
        vocab_q = q.shape[0]

        for j in range(0, vocab_k, 100):
            maxi = min(vocab_k, j + 100)
            size = maxi - j

            # sim = torch.einsum("ke,qe -> kq", k[j:maxi], q)
            # ct_k += (torch.argmax(sim, dim=1) == torch.arange(j, maxi, 1).to(DEVICE)).sum()
            # diag = sim.diagonal(offset=j).reshape(size, 1).expand(size, vocab_q)
            # avg_k += (sim > diag).sum()
            # all_k.extend((sim > diag).sum(dim=1).tolist())

            sim = torch.einsum("qe,ke -> qk", q[j:maxi], k)
            ct_q += (torch.argmax(sim, dim=1) == torch.arange(j, maxi, 1).to(DEVICE)).sum()
            diag = sim.diagonal(offset=j).reshape(size, 1).expand(size, vocab_q)
            avg_q += (sim > diag).sum()
            all_q.extend((sim > diag).sum(dim=1).tolist())

        # axs[head][0].hist(all_k, bins=1000, range=(0, vocab_k))
        axs[head].set_title(f"head 0.{head}")
        axs[head].hist(all_q, bins=100, range=(0, vocab_k))

        print(
            head,
            f"{ct_k / vocab_k:<10.3%} {ct_q / vocab_k:<10.3%} {avg_k / vocab_k:<10.3f} {avg_q / vocab_k:<10.3f}",
        )

    plt.xlabel("Duplicate token rank")
    plt.ylabel("Histogram count")
    plt.show()


def pqpk():
    plt.rcParams["figure.figsize"] = (30, 5)
    fig, axs = plt.subplots(1, 8)
    for head in range(8):
        k = transform_vocab(0, head, "k", normalise=False, pos=True)[:50]
        q = transform_vocab(0, head, "q", normalise=False, pos=True)[:50]
        sim = torch.einsum("ke,qe -> qk", k, q).tril()
        sim -= (torch.ones_like(sim) * 10000).triu(diagonal=1)
        print(sim.shape)
        sim = torch.softmax(sim, dim=1)
        axs[head].imshow(sim.cpu())
        axs[head].set_title(f"0.{head}")
    plt.show()


def umaps():
    plt.rcParams["figure.figsize"] = (5, 5)
    vs: list[torch.Tensor] = []
    for head in tqdm(range(8)):
        v = transform_vocab(0, head, "v", normalise=True, pos=False)
        vs.append(v)
    vs.append(normalised_lnormed)
    g = torch.zeros((9, 9))
    for i in range(9):
        for j in range(i, 9):
            dot = torch.einsum("ve,we -> vw", vs[i], vs[j])
            cos = dot.mean().item()
            print(i, j, cos)
            g[i][j] = cos
            g[j][i] = cos
    plt.imshow(g, vmin=-1, vmax=1, cmap="RdBu")
    plt.yticks(list(range(9)), labels=[f"0.{x}" for x in range(8)] + ["emb"])
    plt.xticks(list(range(9)), labels=[f"0.{x}" for x in range(8)] + ["emb"])
    plt.colorbar()
    plt.show()
    embed_plot_many(*vs, title="v", pos=False, save=True)


def main():
    # embed_plot_many(lnormed_embeds, transform_vocab(0, 0, "v", normalise=True, pos=False), title=f"embeds")
    umaps()


if __name__ == "__main__":
    main()
# %%
