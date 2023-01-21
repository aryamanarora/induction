# %%
import torch

import rust_circuit as rc
from interp.circuit.interop_rust.module_library import load_model_id

import plotly.express as px

DEVICE = "cuda:0"
seq_len = 300

torch.set_grad_enabled(False)

model_id = "attention_only_2"
(loaded, tokenizer, extra_args) = load_model_id(model_id)
loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE)) for (s, c) in loaded.items()}

orig_circuit = loaded["t.bind_w"]
tok_embeds = loaded["t.w.tok_embeds"].value
pos_embeds = loaded["t.w.pos_embeds"].value
all_toks = tokenizer.batch_decode(torch.arange(tok_embeds.shape[0]))

wk = loaded["t.bind_w"].get_unique("a0.w.k_arr").value[0]
wq = loaded["t.bind_w"].get_unique("a0.w.q_arr").value[0]
wv = loaded["t.bind_w"].get_unique("a0.w.v_arr").value[0]

mean_subbed = tok_embeds - tok_embeds.mean(dim=1, keepdim=True)
vars = tok_embeds.var(dim=1, keepdim=True) + 1e-05
denom = torch.sqrt(vars)
ln_bias = loaded["t.bind_w"].get_unique("a0.ln.w.bias_arr").value
ln_scale = loaded["t.bind_w"].get_unique("a0.ln.w.scale_arr").value
naive_lnormed_embeds = mean_subbed / denom
lnormed_embeds = (naive_lnormed_embeds * ln_scale) + ln_bias

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
# %%
