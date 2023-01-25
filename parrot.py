from utils import *

exps = ["unscrubbed", "k-1.5-0.0"]
file = "attn_scores"
inps = load_inputs()[:, :-1]
tokenizer = load_tokenizer()
exps_unwrapped = []
for i in exps:
    if isinstance(i, tuple):
        exps_unwrapped.extend(list(i))
    else:
        exps_unwrapped.append(i)

common_ixes = list(get_common_saa_ixes({exp for exp in exps_unwrapped}, ix_filter=None, type=file))
all_attns = []

for ix in common_ixes:
    all_attns_idx = []
    for exp in exps:
        print(exp)
        if isinstance(exp, tuple):
            with open(f"{RESULTS_PATH}/{exp[0]}_{file}_{ix}.pkl", "rb") as f:
                res1, _, _ = pickle.load(f)  # res1 shape is [batch, heads, q, k]
            with open(f"{RESULTS_PATH}/{exp[1]}_{file}_{ix}.pkl", "rb") as f:
                res2, _, _ = pickle.load(f)  # res1 shape is [batch, heads, q, k]
            all_attns_idx.append(res1.tril() - res2.tril())
        else:
            with open(f"{RESULTS_PATH}/{exp}_{file}_{ix}.pkl", "rb") as f:
                res1, _, _ = pickle.load(f)  # res1 shape is [batch, heads, q, k]
                all_attns_idx.append(res1.tril())
    all_attns.append(torch.cat(all_attns_idx, dim=0))

a = 1 / torch.arange(300, 0, -1)
for i, attns in enumerate(all_attns):
    b = tokenizer.batch_decode(inps[common_ixes[i]])[:-1]
    tops = []
    for j in range(attns.shape[0]):
        attn_15: torch.Tensor = attns[j][13]
        keys = attn_15.sum(dim=0) * a
        tops.append((keys, torch.sort(keys, descending=True).indices))
    print("-" * 50)
    print(common_ixes[i])
    for j in range(10):
        for keys, top in tops:
            print(f"{b[top[j]].replace(' ', '_'):<20}{keys[top[j]].item():<20.3f}", end="")
        print()
