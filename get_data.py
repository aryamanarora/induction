from experiments import make_make_corr, make_experiments, FixedSampler
from main import run_experiment
from tqdm import tqdm

for i in tqdm(range(50)):
    experiments = make_experiments(make_make_corr(FixedSampler(i)))
    for exp in ["0.0", "0.6", "unscrubbed"]:
        save_name = f"{exp}_attns_{i}"
        run_experiment(experiments, exp, 1000, save_name, 0, get_attns=True)
