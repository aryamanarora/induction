# %%
import glob
import torch
import pickle
import os.path
import rust_circuit as rc
from colorama import Fore, Back, Style
from typing import Any, Callable

from interp import cui
from interp.ui.very_named_tensor import VeryNamedTensor
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.rrfs import RRFS_DIR

DEVICE = "cuda"
RESULTS_PATH = "results"
DATA_PATH = "data"


def pickle_tokenizer():
    _, tokenizer, _ = load_model_id("attention_only_2")
    with open(os.path.join(DATA_PATH, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    return tokenizer


def load_tokenizer():
    try:
        with open(os.path.join(DATA_PATH, "tokenizer.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return pickle_tokenizer()


def pickle_inputs():
    P = rc.Parser()
    toks_int_values = P("'toks_int_var' [104091,301] Array 3f36c4ca661798003df14994")
    toks_int_values = rc.cast_circuit(toks_int_values, rc.TorchDeviceDtypeOp(device=DEVICE, dtype="int64")).cast_array()
    toks_indices = torch.arange(toks_int_values.shape[0], device=DEVICE).reshape(-1, 1)
    toks_int_values = torch.concat([toks_int_values.cast_array().value, toks_indices], dim=1)

    with open(os.path.join(DATA_PATH, "full_inps.pkl"), "wb") as f:
        pickle.dump(toks_int_values, f)
    return toks_int_values


def load_inputs():
    """
    Return the entire 104091 x 302 dataset. This includes both the [BEGIN] token at the start
    of each example, as well as the index token at the end of each example. For most uses, you
    will want to trim off the index column at the end, and then either trim the first column
    (to get the labels of the next-token-prediction task) or trim the (new) last column (to get
    the inputs of the next-token-prediction task).
    """
    try:
        with open(os.path.join(DATA_PATH, "full_inps.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return pickle_inputs()


def build_common_toks_order(apply_laplace_smoothing=True):
    """
    Return a 1D tensor containing all token values in the dataset, sorted by descending
    frequency order.
    If apply_laplace_smoothing is True (default), the return tensor includes token values
    not present in the dataset.
    """
    inps = load_inputs()[:, :-1]
    if apply_laplace_smoothing:
        inps = torch.cat((inps.flatten(), torch.arange(inps.max().item() + 1, device=DEVICE)))
    uniq, counts = torch.unique(inps, return_counts=True)
    sort_ixes = torch.argsort(counts, descending=True)
    return uniq[sort_ixes]


def replace_toks_by_frequency_rank(toks):
    """
    toks is a tensor of token values.
    Return a tensor like toks, but where each token value is replaced by its position
    in the order of most common tokens (e.g the most common token value is replaced by 0).
    """
    common_toks = build_common_toks_order()
    return torch.take(torch.argsort(common_toks), toks)


def load_good_induction_candidates():
    cache_dir = f"{RRFS_DIR}/ryan/induction_scrub/cached_vals"
    return torch.load(f"{cache_dir}/induction_candidates_2022-10-15 04:48:29.970735.pt").to(
        device=DEVICE, dtype=torch.float32
    )


def tok_stdize_simple_strip(all_tok_strs, tok_ix):
    return all_tok_strs[tok_ix].upper().strip(" ()[]{},.:;-_\"")


def decode_and_highlight(seq_to_decode, highlight_mask):
    tokenizer = load_tokenizer()
    res = []
    for i, ch in enumerate(seq_to_decode):
        if i == 300:
            res.append(ch)
            break
        if highlight_mask[i]:
            res += [58, 58, 58]
            res.append(ch)
            res += [60, 60, 60]
        else:
            res.append(ch)
    res = "".join(tokenizer.batch_decode(res))
    res = res.replace("]]][[[", f"{Style.RESET_ALL}{Fore.RED}|{Style.RESET_ALL}{Back.RED}")
    res = res.replace("[[[", Back.RED)
    res = res.replace("]]]", f"{Style.RESET_ALL}")
    return res


def get_common_saa_ixes(exps, ix_filter=None, type="saa"):
    exp_fns = [glob.glob(f"{RESULTS_PATH}/{exp}_{type}_*") for exp in exps]
    exp_ixes = [{int(fn.split("_")[-1].split(".")[0]) for fn in fns} for fns in exp_fns]
    if ix_filter is not None:
        exp_ixes.append(set(ix_filter))

    return set.intersection(*exp_ixes)


def get_diff_mean_per_tok_loss(exp1, exp2, inp_ix):
    with open(f"{RESULTS_PATH}/{exp1}_saa_{inp_ix}.pkl", "rb") as f:
        res1, _, _ = pickle.load(f)
    with open(f"{RESULTS_PATH}/{exp2}_saa_{inp_ix}.pkl", "rb") as f:
        res2, _, _ = pickle.load(f)
    return torch.cat(
        (torch.zeros((1, 1)), (res2.mean(dim=0, keepdim=True).cpu() - res1.mean(dim=0, keepdim=True).cpu())),
            dim=1
    )


# Copied from remix_utils
def await_without_await(func: Callable[[], Any]):
    """We want solution files to be usable when run as a script from the command line (where a top level await would
    cause a SyntaxError), so we can do CI on the files. Avoiding top-level awaits also lets us use the normal Python
    debugger.
    Usage: instead of `await cui.init(port=6789)`, write `await_without_await(lambda: cui.init(port=6789))`
    """
    try:
        while True:
            func().send(None)
    except StopIteration:
        pass


def compare_attns_in_cui(exps, ix_filter=None):
    inps = load_inputs()[:, :-1]
    tokenizer = load_tokenizer()
    exps_unwrapped = []
    for i in exps:
        if isinstance(i, tuple):
            exps_unwrapped.extend(list(i))
        else:
            exps_unwrapped.append(i)

    common_ixes = list(get_common_saa_ixes({exp for exp in exps_unwrapped}, ix_filter=ix_filter, type="attns"))
    all_attns = []
    print(exps)
    for ix in common_ixes:
        all_attns_idx = []
        for exp in exps:
            print(exp)
            if isinstance(exp, tuple):
                with open(f"{RESULTS_PATH}/{exp[0]}_attns_{ix}.pkl", "rb") as f:
                    res1, _, _ = pickle.load(f)  # res1 shape is [batch, heads, q, k]
                with open(f"{RESULTS_PATH}/{exp[1]}_attns_{ix}.pkl", "rb") as f:
                    res2, _, _ = pickle.load(f)  # res1 shape is [batch, heads, q, k]
                all_attns_idx.append(res1 - res2)
            else:
                with open(f"{RESULTS_PATH}/{exp}_attns_{ix}.pkl", "rb") as f:
                    res1, _, _ = pickle.load(f)  # res1 shape is [batch, heads, q, k]
                    all_attns_idx.append(res1)
        all_attns.append(torch.cat(all_attns_idx, dim=0))

    comparison_names = [f"Attention in {exp}" for exp in exps]
    vnts = []
    for i, attns in enumerate(all_attns):
        b = tokenizer.batch_decode(inps[common_ixes[i]])[:-1]
        vnts.append(
            VeryNamedTensor(
                attns,
                dim_names=["comparison", "heads", "q", "k"],
                dim_types=["example", "heads", "seq", "seq"],
                dim_idx_names=[comparison_names, [f"a{i // 8}.{i % 8}" for i in range(16)], b, b],
                title=f"Example {common_ixes[i]}",
            )
        )

    await_without_await(lambda: cui.init(port=6789))
    await_without_await(lambda: cui.show_tensors(*vnts))


def compare_saa_in_cui(comparisons, ix_filter=None, mask=None):
    """
    comparisons is an iterable of pairs (exp1, exp2). We compute the diff for each pair,
    for all example ixes on which we have all the relevant exps data.
    If ix_filter is provided, we use only those ixes, rather than all the common ixes.
    """
    inps = load_inputs()[:, :-1]
    tokenizer = load_tokenizer()
    all_exps = {exp for exp_pair in comparisons for exp in exp_pair}
    common_ixes = list(get_common_saa_ixes(all_exps, ix_filter))
    all_loss_diffs = []
    for ix in common_ixes:
        ix_loss_diffs = []
        for exp1, exp2 in comparisons:
            loss_diff = get_diff_mean_per_tok_loss(exp1, exp2, ix)
            if mask is not None:
                loss_diff = loss_diff * mask[ix]
            ix_loss_diffs.append(loss_diff)
        all_loss_diffs.append(torch.cat(ix_loss_diffs, dim=0))

    comparison_names = [f"Loss increase from {exp1} to {exp2}" for (exp1, exp2) in comparisons]

    vnts = []
    for i, ix_diffs in enumerate(all_loss_diffs):
        vnts.append(
            VeryNamedTensor(
                ix_diffs,
                dim_names=["comparison", "pos"],
                dim_types=["example", "seq"],
                dim_idx_names=[comparison_names, tokenizer.batch_decode(inps[common_ixes[i]])],
                title=f"Example {common_ixes[i]}",
            )
        )

    await_without_await(lambda: cui.init(port=6789))
    await_without_await(lambda: cui.show_tensors(*vnts))


# %%
