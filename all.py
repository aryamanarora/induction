from experiments import make_experiments
from main import run_experiment
from tqdm import tqdm

experiments = make_experiments()
print(len(experiments))
for exp in tqdm(experiments):
    run_experiment(experiments, exp, 10000)
