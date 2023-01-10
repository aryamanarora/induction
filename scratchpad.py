import torch
import pickle
import plotly.express as px
from colorama import Fore, Back, Style

def load_res(res_path):
    with open(f"data/{res_path}.pkl", "rb") as f:
        return pickle.load(f)

def load_all(res_path):
    r, m1, m2 = load_res(res_path)
    with open(f"data/inps_{res_path}.pkl", 'rb') as f:
        i = pickle.load(f)
    return r, m1, m2, i

def get_diff_ixes(inp1, inp2):
    diff_ixes = []
    for i in range(inp1.shape[0]):
        if not (inp1[i] == inp2[i]).all():
            diff_ixes.append(i)
    return diff_ixes


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

def compare_losses(exp1, exp2, tokenizer, inps):
    res_1, ind_candidates_mask_1, ind_candidates_later_occur_mask_1 = load_res(exp1)
    res_2, ind_candidates_mask_2, ind_candidates_later_occur_mask_2 = load_res(exp2)
    exp1_better = res_1 < res_2
    for i in range(res_1.shape[0]):
        print(decode_and_highlight(inps[i], ind_candidates_later_occur_mask_1[i], tokenizer))
        print(decode_and_highlight(inps[i], exp1_better[i], tokenizer))
        inp = input()
        if inp == 'q':
            break
        else:
            pass

def build_hist(exp1, exp2):
    res_1, ind_candidates_mask_1, ind_candidates_later_occur_mask_1 = load_res(exp1)
    res_2, ind_candidates_mask_2, ind_candidates_later_occur_mask_2 = load_res(exp2)
    data = res_1[ind_candidates_later_occur_mask_1] - res_2[ind_candidates_later_occur_mask_1]
    fig = px.histogram(torch.flatten(data.cpu()), range_y=[0, 3000], 
        range_x=[-10, 3])
    fig.write_image("other_img.jpg")

with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("data/inps.pkl", "rb") as f:
    inps = pickle.load(f)

    
build_hist("unscrubbed", "baseline")
compare_losses("a0-v-0", "not-ev", tokenizer, inps)