# %%
# This is a demo and explainer on how to use the Subgraph Ablation Attribution
# CUI visualizations.

import pickle
import os.path
from utils import compare_saa_in_cui, compare_attns_in_cui
from masks import get_all_masks

DATA_PATH = "data"
RESULTS_PATH = "results"

compare_attns_in_cui(
    ["real-0.0"],
    "attns",
)
# %%

compare_attns_in_cui(
    [
        ("unscrubbed", "k-1.5-0.0"),
        ("unscrubbed", "k-1.6-0.0"),
    ],
    "attn_scores",
)
