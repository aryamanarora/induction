# %%
import torch

import rust_circuit as rc
from interp.circuit.interop_rust.module_library import load_model_id
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('l')
parser.add_argument('h')
args = parser.parse_args()

DEVICE = "cuda:0"
HEAD = (int(args.l), int(args.h))
seq_len = 300

COMPUTE_QKS = False
COMPUTE_OV_COS_SIMS = True

torch.set_grad_enabled(False)

model_id = "attention_only_2"
(loaded, tokenizer, extra_args) = load_model_id(model_id)
loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE)) for (s, c) in loaded.items()}

def get_top_token_strs_and_vals(qk_matrix, top_k, largest=True, decode_q=True, decode_k=True):
    q_func = tokenizer.decode if decode_q else lambda x: x.item()
    k_func = tokenizer.decode if decode_k else lambda x: x.item()
    top_vals, top_ixes = torch.topk(qk_matrix.flatten().cpu(), top_k, largest=largest)
    top_toks = [(q_func(ix // qk_matrix.shape[1]), k_func(ix % qk_matrix.shape[1])) for ix in top_ixes]
    return top_vals, top_toks

orig_circuit = loaded["t.bind_w"]
tok_embeds = loaded["t.w.tok_embeds"].value
pos_embeds = loaded["t.w.pos_embeds"].value

wk = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.w.k_arr").value[HEAD[1]]
wq = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.w.q_arr").value[HEAD[1]]
wv = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.w.v_arr").value[HEAD[1]]
wo = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.w.o_arr").value[HEAD[1]]

ln_bias = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.ln.w.bias_arr").value
ln_scale = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.ln.w.scale_arr").value
lnormed_embeds = torch.nn.functional.layer_norm(tok_embeds, (tok_embeds.shape[1],), ln_scale, ln_bias)
del ln_bias
del ln_scale
del tok_embeds
torch.cuda.empty_cache()

if COMPUTE_OV_COS_SIMS:
    lnormed_norms = torch.norm(lnormed_embeds, dim=1)
    lnormed_cos = (torch.einsum("ve,we -> vw", lnormed_embeds, lnormed_embeds) / lnormed_norms.unsqueeze(0)) / lnormed_norms.unsqueeze(1)
    tok_os = torch.einsum("va,ea -> ve", torch.einsum("ve,ae -> va", lnormed_embeds, wv), wo)
    o_norms = torch.norm(tok_os, dim=1)
    output_cos = (torch.einsum("ve,we -> vw", tok_os, tok_os) / o_norms.unsqueeze(0)) / o_norms.unsqueeze(1)
    del o_norms
    del tok_os
    del lnormed_norms
    torch.cuda.empty_cache()
    cos_increase = (output_cos - lnormed_cos)
    cos_increase = cos_increase.tril()
    lnormed_cos = torch.nn.functional.relu(lnormed_cos)
    weighted_cos_increase = cos_increase * lnormed_cos

if COMPUTE_QKS:
    torch.cuda.empty_cache()
    tok_ks = torch.einsum("ve,ae -> va", lnormed_embeds, wk)
    tok_qs = torch.einsum("ve,ae -> va", lnormed_embeds, wq)
    pos_ks = torch.einsum("ve,ae -> va", pos_embeds, wk)[:350]
    pos_qs = torch.einsum("ve,ae -> va", pos_embeds, wq)[:350]

    tqtk = torch.einsum("va,wa -> vw", tok_qs, tok_ks)
    pqpk = torch.einsum("pa,qa -> pq", pos_qs, pos_ks).tril()
    tqpk = torch.einsum("va,pa -> vp", tok_qs, pos_ks)
    pqtk = torch.einsum("pa,va -> pv", pos_qs, tok_ks)

torch.cuda.empty_cache()
top_vals, top_ixes = torch.topk(weighted_cos_increase.flatten().cpu(), 3000, largest=True)
top_toks = [((ix // weighted_cos_increase.shape[1]).item(), (ix % weighted_cos_increase.shape[1]).item()) for ix in top_ixes]
toks = [(tokenizer.decode(fst), tokenizer.decode(snd)) for fst, snd in top_toks]
top_increases = [round(cos_increase[fst, snd].item(), 5) for fst, snd in top_toks]
top_origs = [round(lnormed_cos[fst, snd].item(), 5) for fst, snd in top_toks]
to_write = zip(toks, top_increases, top_origs)

with open(f"{HEAD[0]}{HEAD[1]}_similar_originally_similar.tsv", "w", newline='') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')
    for line in to_write:
        writer.writerow(line)
# %%
