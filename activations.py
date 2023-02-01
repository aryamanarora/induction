import torch
import rust_circuit as rc
import pickle
import utils
import os.path
import plotly.express as px

from tqdm import tqdm

from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.indexer import TORCH_INDEXER as I

# Most of these should be handled with argparse. Probably should pre-compute
# most matchers and have that as a command-line argument also.
MODE = "compose"
DEVICE = "cuda:0"
SEQ_LEN = 300
NUM_SAMPLES = 10000
BATCH_SIZE = 100
MATCHER = rc.IterativeMatcher("b1.call").chain(
    rc.restrict("b.a", end_depth=2)).chain(
    rc.restrict("a.q", end_depth=7)).chain(
    rc.restrict("b0.call", end_depth=8))
PRE_TRANSFORM = lambda t: t
POST_TRANSFORM = lambda l: torch.concat(l)
SAVE_PATH = "data/activations/l1_pre_ln_10000.pkl"

L0_HEAD = 0
L1_HEAD = 0

torch.set_grad_enabled(False)

# Plausibly the best version of this is a matcher parameter you pass onto
# main.py:run_experiment that saves the output of that parameter, in particular
# because this would allow for saving activations from scrubbed runs.
# Also: this takes up a lot of disk (~3GB for 10k samples) and is not slow to
# compute at all. Probably best to not pickle them.
def collect_activations(num_samples, matcher, pre_transform, post_transform, save_path):
    """
    Collect activations from the unscrubbed model.
    num_samples is how many samples to collect the activations for
    matcher is a matcher uniquely specifying a specific node in the unscrubbed model, post
    `rc.substitute_all_modules`
    pre_transform is a function that takes in the result of calling `evaluate` on the
    matched node and returns the tensor you want to save from that run (e.g. selecting the
    activations from a specific head)
    post_transform is a function that takes in a list of outputs of pre_transform and returns
    the tensor to save (e.g. concatting)
    save_path is the path in which to save the result of post_transform
    """
    model_id = "attention_only_2"
    (loaded, tokenizer, extra_args) = load_model_id(model_id)

    P = rc.Parser()
    toks_int_values = P("'toks_int_var' [104091,301] Array 3f36c4ca661798003df14994")
    toks_int_values = rc.cast_circuit(toks_int_values, rc.TorchDeviceDtypeOp(device=DEVICE, dtype="int64")).cast_array()
    toks_indices = torch.arange(toks_int_values.shape[0], device=DEVICE).reshape(-1, 1)
    toks_int_values = rc.Array(
        torch.concat([toks_int_values.cast_array().value, toks_indices], dim=1),
        name="toks_int_var",
    )
    loaded = {s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device=DEVICE)) for (s, c) in loaded.items()}

    orig_circuit = loaded["t.bind_w"]
    tok_embeds = loaded["t.w.tok_embeds"]
    pos_embeds = loaded["t.w.pos_embeds"]

    samples_collected = 0
    to_save = []
    with tqdm(total=num_samples) as pbar:
        while samples_collected < num_samples:
            # input
            input_toks = toks_int_values.index(I[samples_collected:samples_collected + BATCH_SIZE, :-2], name="input_toks_int")

            # feed input tokens to model (after embedding + causal mask)
            idxed_embeds = rc.GeneralFunction.gen_index(tok_embeds, input_toks, index_dim=0, name="idxed_embeds")
            causal_mask = rc.Array(
                (torch.arange(SEQ_LEN)[:, None] >= torch.arange(SEQ_LEN)[None, :]).to(tok_embeds.cast_array().value),
                f"t.a.c.causal_mask",
            )
            pos_embeds = pos_embeds.index(I[:SEQ_LEN], name="t.w.pos_embeds_idxed")
            model = rc.module_new_bind(
                orig_circuit, ("t.input", idxed_embeds), ("a.mask", causal_mask), ("a.pos_input", pos_embeds), name="t.call"
            )
            model = rc.substitute_all_modules(model)
            circuit_to_eval = model.get_unique(matcher)
            to_save.append(pre_transform(circuit_to_eval.evaluate()))
            samples_collected += BATCH_SIZE
            pbar.update(BATCH_SIZE)
            torch.cuda.empty_cache()
    with open(save_path, "wb") as f:
        pickle.dump(post_transform(to_save), f)


def compute_cross_layer_composition(l0_head, l1_head):
    wk, _, _, _ = utils.load_attn_weights(1, l1_head)
    with open(os.path.join(utils.DATA_PATH, "activations", f"0.{l0_head}_ov_10000.pkl"), "rb") as f:
        real_ovs_0x = pickle.load(f).to(utils.DEVICE)
    with open(os.path.join(utils.DATA_PATH, "activations", f"1.{l1_head}_q_10000.pkl"), "rb") as f:
        real_q_1x = pickle.load(f).to(utils.DEVICE)
    with open(os.path.join(utils.DATA_PATH, "activations", "l1_pre_ln_10000.pkl"), "rb") as f:
        real_preln_l1 = pickle.load(f).to(utils.DEVICE)
    real_preln_l1 -= real_ovs_0x
    real_q_1x = torch.einsum("bpa,ae->bpe", real_q_1x, wk)
    del wk
    del real_ovs_0x

    _, _, wv, wo = utils.load_attn_weights(0, l0_head)
    torch.cuda.empty_cache()

    loaded = utils.load_model()
    ln_1_scale = loaded["t.bind_w"].get_unique("a1.ln.w.scale_arr").value
    ln_1_bias = loaded["t.bind_w"].get_unique("a1.ln.w.bias_arr").value
    tok_embeds, _ = utils.load_embeds()
    tok_embeds = utils.layer_norm(tok_embeds, 0)
    tok_embeds = torch.einsum("ve,ae->va", tok_embeds, wv)
    tok_embeds = torch.einsum("va,ea->ve", tok_embeds, wo)
    del _
    del wv
    del wo
    del loaded
    torch.cuda.empty_cache()

    means = []
    for embed in tqdm(tok_embeds):
        post_ln = torch.nn.functional.layer_norm(real_preln_l1 + embed, (embed.shape[-1],), ln_1_scale, ln_1_bias)
        means.append(torch.einsum("bpe,bre->bpr", real_q_1x, post_ln).mean())
        torch.cuda.empty_cache()

    with open(f"data/activations/mean_0{l0_head}_1{l1_head}_compositions.pkl", "wb") as f:
        pickle.dump(means, f)


if __name__ == "__main__":
    if MODE == "collect":
        collect_activations(NUM_SAMPLES, MATCHER, PRE_TRANSFORM, POST_TRANSFORM, SAVE_PATH)
    elif MODE == "compose":
        compute_cross_layer_composition(L0_HEAD, L1_HEAD)
    else:
        print("No valid mode specified")