import glob
import pickle
import torch
from masks import get_all_masks
from utils import consolidate_results

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def get_mean_losses(exp_name, sample_ix):
    """
    Get mean losses from SAA experiments.
    """
    with open(f"results/{exp_name}_saa_{sample_ix}.pkl", "rb") as f:
        res, _, _ = pickle.load(f)
    return res.mean(dim=0)


def get_loss_tensors_and_ixes(exp_name):
    """
    Get loss tensors for an experiment as well as for the same ixes for unscrubbed
    Also return the input indices.
    """
    exp_fns = sorted(glob.glob(f"results/{exp_name}_saa_*.pkl"))
    inp_ixes = [int(fn.split("_")[-1].split(".")[0]) for fn in exp_fns]
    losses = torch.stack([get_mean_losses(exp_name, ix) for ix in inp_ixes])
    unscrubbed_losses = torch.stack([get_mean_losses("unscrubbed", ix) for ix in inp_ixes])
    return losses, unscrubbed_losses, inp_ixes


def plot_loss_diff_histogram(exp_name, mask_names=None, ceil=1.0):
    """
    Plot histograms of the differences between an experiment's mean per-token losses
    and the unscrubbed per-token losses.
    If mask_names is provided, mask the results before plotting for each mask_name
    ceil is the maximum y-axis value.
    """
    consolidate_results({exp_name})
    losses, unscrubbed_losses, inp_ixes = get_loss_tensors_and_ixes(exp_name)
    losses = losses - unscrubbed_losses
    all_masks = get_all_masks(inp_ixes)
    all_masks["all"] = torch.ones_like(losses, dtype=bool)
    masks = [all_masks[mask_name] for mask_name in mask_names]
    all_losses = [losses.flatten().cpu()] + [losses[mask].cpu() for mask in masks]
    mask_names = ["all"] + mask_names
    fig = fig = make_subplots(rows=len(mask_names), cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for i, mask_name in enumerate(mask_names):
        data = losses[all_masks[mask_name]].cpu()
        trace = go.Histogram(x=data, nbinsx=200, name=mask_name, histnorm="probability")
        fig.add_trace(trace, row=i+1, col=1)
        fig.update_yaxes(range=[0, ceil], row=i+1, col=1)
        fig.update_xaxes(range=[-8, 8], row=i+1, col=1)
        
    fig.update_layout(title=f"Distribution of per-token loss differences for {exp_name}", height=800)
    fig.write_image("tmp.png")
    fig.show()


def plot_loss_scatterplot(exp_name, mask_name=None):
    """
    Plot a scatter plot of per-token original loss vs scrubbed loss.
    If mask is provided, mask the results before plotting.
    """
    losses, unscrubbed_losses, inp_ixes = get_loss_tensors_and_ixes(exp_name)
    mask = mask or torch.ones_like(losses)
    if mask_name is not None:
        mask = get_all_masks(inp_ixes)[mask_name]
    losses = losses[mask]
    unscrubbed_losses = unscrubbed_losses[mask]
    fig = px.scatter(x=unscrubbed_losses.cpu().numpy(), y=losses.cpu().numpy(), title=f"Scatterplot of per-token losses for {exp_name} against unscrubbed")

    # Center at y-origin
    xlim = unscrubbed_losses.max()
    ylim = torch.abs(losses).max()
    fig.update_layout(xaxis_range=[0, xlim], yaxis_range=[-ylim, ylim])

    # Add cartesian axes lines
    fig.add_shape(type='line',
              x0=0, y0=fig.layout.yaxis.range[0],
              x1=0, y1=fig.layout.yaxis.range[1],
              line=dict(color='black', width=2))
    fig.add_shape(type='line',
              x0=fig.layout.xaxis.range[0], y0=0,
              x1=fig.layout.xaxis.range[1], y1=0,
              line=dict(color='black', width=2))
    
    fig.show()

if __name__ == "__main__":
    plot_loss_diff_histogram("positional-all-naive-with-multi-tok-ind", ["not_uncommon_repeats", "uncommon_repeats"], ceil=0.01)