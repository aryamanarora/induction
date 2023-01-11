import torch
import pickle
import rust_circuit as rc
import plotly.express as px
from colorama import Fore, Back, Style

DEVICE='cuda'

def load_res(res_path):
    with open(f"data/{res_path}.pkl", "rb") as f:
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