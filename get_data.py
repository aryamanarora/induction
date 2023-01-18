from experiments import *
from tqdm import tqdm

for i in tqdm(range(50)):
    experiments = make_experiments(make_make_corr(FixedSampler(i)))
    for exp in [f"1.{j}" for j in [0, 5, 6]] + ["unscrubbed"]:
        save_name = f"{exp}_saa_{i}"
        run(experiments, exp, 1000, save_name, 0)
