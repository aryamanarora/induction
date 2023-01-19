# %%
# This is a demo and explainer on how to use the Subgraph Ablation Attribution
# CUI visualizations.

import pickle
import os.path
from utils import compare_saa_in_cui, compare_attns_in_cui
from masks import get_all_masks

DATA_PATH = "data"
RESULTS_PATH = "results"

# The first step is to ensure you have all the pickles you need for the
# visualization. This generally means running experiments.py with an --idx
# argument and the --save flag.
#
# Suppose you want to compare how important the eq and ev paths are. Then you
# might run the following experiments:
#
# python experiments.py --samples 1000 --exp unscrubbed --idx 0 --save
# python experiments.py --samples 1000 --exp eq --idx 0 --save
# python experiments.py --samples 1000 --exp ev --idx 0 --save
#
# At this point, your results directory should be populated with the following:
#
# unscrubbed_saa_0.pkl
# eq_saa_0.pkl
# ev_saa_0.pkl
#
# Now you can specify the comparisons you want to see in the CUI. Below, we are
# trying to see the loss increase from unscrubbed to eq, the loss increase from
# unscrubbed to ev, and the loss increase from ev to eq:

comparisons = [
    ("unscrubbed", "1.5"),
    ("unscrubbed", "1.0"),
    ("unscrubbed", "1.6"),
]

# Now we can open CUI. At the top, you should be able to choose any dataset
# example for which we have all the relevant experiment pickles (in the case
# above, this would be only 0. If you have also run the same three commands for
# idx 1, you would also be able to choose 1. If you have run only one or two of
# those 3 commands for index 2, 2 would not show up). As the view, you should
# have Comparison (example) set to facet, and Pos(seq) set to axis. The chart
# type should be Colored Text.

# compare_saa_in_cui(comparisons)

compare_attns_in_cui(["0.0", "unscrubbed"])
# %%
