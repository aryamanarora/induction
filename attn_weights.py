# %%
import torch

import rust_circuit as rc
from interp.circuit.interop_rust.module_library import load_model_id

import plotly.express as px

DEVICE = "cuda:0"
HEAD = (0, 0)
seq_len = 300

torch.set_grad_enabled(False)

model_id = "attention_only_2"
(loaded, tokenizer, extra_args) = load_model_id(model_id)
loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE)) for (s, c) in loaded.items()}

orig_circuit = loaded["t.bind_w"]
tok_embeds = loaded["t.w.tok_embeds"].value
pos_embeds = loaded["t.w.pos_embeds"].value
all_toks = tokenizer.batch_decode(torch.arange(tok_embeds.shape[0]))

wk = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.w.k_arr").value[HEAD[1]]
wq = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.w.q_arr").value[HEAD[1]]
wv = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.w.v_arr").value[HEAD[1]]

ln_bias = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.ln.w.bias_arr").value
ln_scale = loaded["t.bind_w"].get_unique(f"a{HEAD[0]}.ln.w.scale_arr").value
lnormed_embeds = torch.nn.functional.layer_norm(tok_embeds, (tok_embeds.shape[1],), ln_scale, ln_bias)

# Linearity of the QK matmuls lets us precompute the dot products such that in the end,
# to get a given attn score, we just need to add up four terms:
# (t+p)(t'+p') = tt' + tp' + t'p + pp'
# But the main point is that we can analyze these independently
tok_ks = torch.einsum("ve,ae -> va", lnormed_embeds, wk)
tok_qs = torch.einsum("ve,ae -> va", lnormed_embeds, wq)
tok_vs = torch.einsum("ve,ae -> va", lnormed_embeds, wv)

pos_ks = torch.einsum("ve,ae -> va", pos_embeds, wk)[:350]
pos_qs = torch.einsum("ve,ae -> va", pos_embeds, wq)[:350]

tqtk = torch.einsum("va,wa -> vw", tok_qs, tok_ks)
pqpk = torch.einsum("pa,qa -> pq", pos_qs, pos_ks).tril()
tqpk = torch.einsum("va,pa -> vp", tok_qs, pos_ks)
pqtk = torch.einsum("pa,va -> pv", pos_qs, tok_ks)

def get_top_token_strs_and_vals(qk_matrix, top_k, largest=True, decode_q=True, decode_k=True):
    q_func = tokenizer.decode if decode_q else lambda x: x.item()
    k_func = tokenizer.decode if decode_k else lambda x: x.item()
    top_vals, top_ixes = torch.topk(qk_matrix.flatten().cpu(), top_k, largest=largest)
    top_toks = [(q_func(ix // qk_matrix.shape[1]), k_func(ix % qk_matrix.shape[1])) for ix in top_ixes]
    return top_vals, top_toks
# %%
