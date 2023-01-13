# %%
import glob
import torch
import pickle
import os.path
import rust_circuit as rc
import plotly.express as px
from colorama import Fore, Back, Style
from typing import Any, Callable

from interp import cui
from interp.ui.very_named_tensor import VeryNamedTensor
from interp.circuit.interop_rust.module_library import load_model_id

DEVICE='cuda'
RESULTS_PATH = 'results'
DATA_PATH = 'data'

def pickle_tokenizer():
    _, tokenizer, _ = load_model_id("attention_only_2")
    with open(os.path.join(DATA_PATH, "tokenizer.pkl"), 'wb') as f:
        pickle.dump(tokenizer, f)


def load_tokenizer():
    with open(os.path.join(DATA_PATH, "tokenizer.pkl"), 'rb') as f:
        return pickle.load(f)


def pickle_inputs():
    P = rc.Parser()
    toks_int_values = P("'toks_int_var' [104091,301] Array 3f36c4ca661798003df14994")
    toks_int_values = rc.cast_circuit(
        toks_int_values, rc.TorchDeviceDtypeOp(device=DEVICE, dtype="int64")
    ).cast_array()
    toks_indices = torch.arange(toks_int_values.shape[0], device=DEVICE).reshape(-1, 1)
    toks_int_values = torch.concat([toks_int_values.cast_array().value, toks_indices], dim=1)

    with open(os.path.join(DATA_PATH, 'full_inps.pkl'), 'wb') as f:
        pickle.dump(toks_int_values, f)


def load_inputs():
    with open(os.path.join(DATA_PATH, 'full_inps.pkl'), 'rb') as f:
        return pickle.load(f)[:, :-1]


def load_res(res_path):
    with open(os.path.join(DATA_PATH, f"{res_path}.pkl"), "rb") as f:
        return pickle.load(f)


def load_all(res_path):
    r, m1, m2 = load_res(res_path)
    with open(f"data/inps_{res_path}.pkl", 'rb') as f:
        i = pickle.load(f)
    return r, m1, m2, i


def decode_and_highlight(seq_to_decode, highlight_mask, tokenizer):
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


def compare_losses(exp1, exp2, tokenizer, inps, threshold=0):
    res_1, ind_candidates_mask_1, ind_candidates_later_occur_mask_1 = load_res(exp1)
    res_2, ind_candidates_mask_2, ind_candidates_later_occur_mask_2 = load_res(exp2)
    performance_diff = res_1 - res_2
    i = 0
    while True:
        mask = performance_diff < (-1 * threshold)
        print(decode_and_highlight(inps[i], ind_candidates_later_occur_mask_1[i], tokenizer))
        print('\n\n')
        print(decode_and_highlight(inps[i], mask[i], tokenizer))
        inp = input()
        if inp == 'q':
            break
        elif "t" in inp:
            threshold = float(inp.split()[-1])
        else:
            i += 1


def build_hist(exp1, exp2):
    res_1, ind_candidates_mask_1, ind_candidates_later_occur_mask_1 = load_res(exp1)
    res_2, ind_candidates_mask_2, ind_candidates_later_occur_mask_2 = load_res(exp2)
    data = res_1[ind_candidates_later_occur_mask_1] - res_2[ind_candidates_later_occur_mask_1]
    fig = px.histogram(torch.flatten(data.cpu()), range_y=[0, 3000],
        range_x=[-10, 3])
    fig.write_image("hist.jpg")


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
    try:
        inps = load_inputs()
    except FileNotFoundError:
        pickle_inputs()
        inps = load_inputs()

    try:
        tokenizer = load_tokenizer()
    except FileNotFoundError:
        pickle_tokenizer()
        tokenizer = load_tokenizer()

    exp11, _ = comparisons[0]
    exp11_files = glob.glob(f'{RESULTS_PATH}/{exp11}_saa_*')
    ixes = [int(fn.split('_')[-1].split('.')[0]) for fn in exp11_files]
    if ix_filter is not None:
        ix_filter = set(ix_filter)
        ixes = [ix for ix in ixes if ix in ix_filter]

    common_ixes = []
    all_loss_diffs = []
    for ix in ixes:
        ix_loss_diffs = []
        try:
            for exp1, exp2 in comparisons:
                with open(f'{RESULTS_PATH}/{exp1}_saa_{ix}.pkl', 'rb') as f:
                    res1, _, _ = pickle.load(f)
                with open(f'{RESULTS_PATH}/{exp2}_saa_{ix}.pkl', 'rb') as f:
                    res2, _, _ = pickle.load(f)
                loss_diff = torch.cat((torch.zeros((1,1)), (res2.mean(dim=0,
                    keepdim=True) - res1.mean(dim=0, keepdim=True)).cpu()), dim=1)
                ix_loss_diffs.append(loss_diff)
            common_ixes.append(ix)
            all_loss_diffs.append(torch.cat(ix_loss_diffs, dim=0))

        except FileNotFoundError:
            pass

    comparison_names = [f"Loss increase from {exp1} to {exp2}" for (exp1, exp2) in comparisons]

    vnts = []
    for i, ix_diffs in enumerate(all_loss_diffs):
        vnts.append(VeryNamedTensor(
            ix_diffs,
            dim_names=["comparison", "pos"],
            dim_types=["example", "seq"],
            dim_idx_names=[comparison_names, tokenizer.batch_decode(inps[common_ixes[i]])],
            title=f"Example {common_ixes[i]}",
        ))

    await_without_await(lambda: cui.init(port=6789))
    await_without_await(lambda: cui.show_tensors(*vnts))
# %%
