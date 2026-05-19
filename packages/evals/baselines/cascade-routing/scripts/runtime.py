from selection import CascadeRouter, BaseQualityComputer, BaseCostComputer
import numpy as np
import pandas as pd

class RandomQualityComputer(BaseQualityComputer):
    def __init__(self):
        self.n_models = 0
        super(RandomQualityComputer, self).__init__()

    def fit(self, questions, model_answers, measure):
        self.n_models = len(model_answers[0])


    def predict(self, questions, model_answers=None):
        variances = [np.eye(self.n_models) * 1 / 100 for _ in range(len(questions))]
        return np.random.rand(len(questions), self.n_models), variances

class RandomCostComputer(BaseCostComputer):
    def __init__(self):
        self.n_models = 0

    def fit(self, questions, model_answers, measure):
        self.n_models = len(model_answers[0])

    def predict(self, questions, model_answers=None):
        return np.random.rand(len(questions), self.n_models)
    
import time
from tqdm import tqdm

def eval_cascade(greedy=False, do_speedup=True, no_expect=False, max_depth=None, n_iterations=100, n_models=20):
    qual_computer = RandomQualityComputer()
    qual_computer.fit(
        ['Hello'],
        [[1] * n_models],
        ['Hello']
    )
    cost_computer = RandomCostComputer()
    cost_computer.fit(
        ['Hello'],
        [[1] * n_models],
        ['Hello']
    )
    router = CascadeRouter(
        models=[i for i in range(n_models)],
        max_expected_cost=10,
        quality_computer=qual_computer,
        cost_computer=cost_computer,
        greedy=greedy,
        do_speedup=do_speedup,
        set_sigma_none=no_expect,
        max_depth=max_depth
    )
    times = []
    for i in tqdm(range(n_iterations)):
        router.lambdas = np.random.rand(n_models)
        router.gamma = 0
        start = time.time()
        router.predict(['Hello'], [[None] * n_models])
        times.append(time.time() - start)
    return times

if __name__ == '__main__':
    output = []
    for models in range(2, 101, 5):
        for approach in ['normal', 'greedy', 'no_expect', 'no_speedup', 'max_depth']:
            if approach == 'greedy':
                res = eval_cascade(greedy=True, do_speedup=True, no_expect=False, n_models=models)
            elif approach == 'no_expect':
                res = eval_cascade(greedy=False, do_speedup=True, no_expect=True, n_models=models)
            elif approach == 'no_speedup' and models <= 17:
                res = eval_cascade(greedy=False, do_speedup=False, no_expect=False, n_models=models)
            elif approach == 'max_depth':
                res = eval_cascade(greedy=False, do_speedup=True, no_expect=False, max_depth=3, n_models=models)
            else:
                res = eval_cascade(greedy=False, do_speedup=True, no_expect=False, n_models=models)
            output.append({
                'models': models,
                'approach': approach,
                'time': np.mean(res),
                "std": np.std(res)
            })
        pd.DataFrame(output).to_csv('data/results/random_timing.csv')