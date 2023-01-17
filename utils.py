# %%
import glob
import torch
import pickle
import os.path
import rust_circuit as rc
import plotly.express as px
from tqdm import tqdm
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
        with open(os.path.join(DATA_PATH, 'full_inps.pkl'), 'rb') as f:
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


def build_token_frequency_filter(top_k):
    inps = load_inputs()[:, :-1]
    inps_by_frequency = replace_toks_by_frequency_rank(inps)
    return inps_by_frequency >= top_k


def load_good_induction_candidates():
    cache_dir = f"{RRFS_DIR}/ryan/induction_scrub/cached_vals"
    return torch.load(f"{cache_dir}/induction_candidates_2022-10-15 04:48:29.970735.pt").to(
        device=DEVICE, dtype=torch.float32
    )


def build_basic_token_filters():
    inps = load_inputs()[:, :-1]
    good_induction_candidates = load_good_induction_candidates().to(dtype=torch.bool)

    candidates_mask = torch.zeros_like(inps, dtype=torch.bool)
    repeats_mask = torch.zeros_like(inps, dtype=torch.bool)
    for i, row in tqdm(enumerate(inps), total=inps.shape[0]):
        seen_toks = set()
        for j, tok in enumerate(row):
            if good_induction_candidates[tok]:
                candidates_mask[i, j] = True
            if tok.item() in seen_toks:
                repeats_mask[i, j] = True
            seen_toks.add(tok.item())
    with open(os.path.join(DATA_PATH, "mask_candidates.pkl"), "wb") as f:
        pickle.dump(candidates_mask, f)
    with open(os.path.join(DATA_PATH, "mask_repeats.pkl"), "wb") as f:
        pickle.dump(repeats_mask, f)
    with open(os.path.join(DATA_PATH, "mask_repeat_candidates.pkl"), "wb") as f:
        pickle.dump(repeats_mask.logical_and(candidates_mask), f)


def build_end_of_repeated_bigram_filter():
    """
    Pickle and return a bool tensor of shape 104091 x 301 (dataset without index
    column) indicating whether the given dataset token is the end of a repeated
    bigram in that example (i.e. whether performing strict induction on the
    previous token would upweigh the given token).
    """
    inps = load_inputs()[:, :-1]
    end_of_repeated_bigram_mask = torch.zeros_like(inps, dtype=torch.bool)
    for i, row in tqdm(enumerate(inps), total=inps.shape[0]):
        fst = row[0].item()
        snd = row[1].item()
        seen_bigrams = set([(fst, snd)])
        for j, tok in enumerate(row[2:-1]):
            fst, snd = snd, tok.item()
            if (fst, snd) in seen_bigrams:
                end_of_repeated_bigram_mask[i, j+2] = True
            else:
                seen_bigrams.add((fst, snd))
    with open(os.path.join(DATA_PATH, "mask_ends_of_repeated_bigrams.pkl"), "wb") as f:
        pickle.dump(end_of_repeated_bigram_mask, f)



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


def get_common_saa_ixes(exps, ix_filter=None):
    exp_fns = [glob.glob(f"{RESULTS_PATH}/{exp}_saa_*") for exp in exps]
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
        (torch.zeros((1, 1)), (res2.mean(dim=0, keepdim=True) - res1.mean(dim=0, keepdim=True)).cpu()), dim=1
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


def compare_saa_in_cui(comparisons, ix_filter=None):
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
