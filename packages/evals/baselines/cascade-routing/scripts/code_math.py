from selection import test_everything, HyperoptStrategy, ConstantStrategy, RepetitiveConstantStrategy
from sklearn.linear_model import LogisticRegression
from loguru import logger
import sys
import os
import pandas as pd
import json
import numpy as np

np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# set logger to only show info messages
logger.remove()
logger.add(sys.stdout, level="INFO")

def strat1(max_lambda):
    return ConstantStrategy(max_lambda=max_lambda, n_iterations=30)

def strat2(max_lambda):
    return RepetitiveConstantStrategy(max_lambda=max_lambda, n_iterations=30)

def strat4(max_lambda):
    return HyperoptStrategy(max_lambda=max_lambda, n_searches=1000, max_factor=4)

def strat3(max_lambda):
    return HyperoptStrategy(max_lambda=max_lambda, n_searches=1000, max_factor=4, optimize_max_depth=True)

all_models = [
        {
            'name': 'Qwen2.5-Math-1.5B-Ins',
        },
        {
            'name': 'Qwen2.5-Coder-1.5B-Ins',
        },
        {
            'name': 'Qwen2.5-Coder-7B-Ins',
        },
        {
            'name': 'Qwen2.5-Math-7B-Ins',
        },
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CodeMath experiments")
    parser.add_argument("--dataset", type=str, default="livecodebench_minerva", help="Dataset name")
    parser.add_argument('--models', type=str, default='0,1,2,3')
    parser.add_argument('--assume-constant-costs', action='store_true')
    parser.add_argument("--half-test", action='store_true')

    args = parser.parse_args()

    dataset = args.dataset
    data_folder = 'data/code_math'

    test_model_answers = pd.read_json(os.path.join(data_folder, dataset, 'test', 'model_answers.json'))
    test_costs = pd.read_json(os.path.join(data_folder, dataset, 'test', 'costs.json'))
    test_qualities = pd.read_json(os.path.join(data_folder, dataset, 'test', 'qualities.json'))
    test_queries = pd.read_json(os.path.join(data_folder, dataset, 'test', 'queries.json'))
    test_queries = [(q[0][0], q[1], q[2]) for i, q in test_queries.iterrows()]
    

    if args.half_test:
        np.random.seed(0)
        train_indices = np.random.choice(len(test_queries), len(test_queries) // 2, replace=False)
        test_indices = np.array([i for i in range(len(test_queries)) if i not in train_indices])
        train_queries = [test_queries[i] for i in train_indices]
        train_model_answers = test_model_answers.iloc[train_indices]
        train_costs = test_costs.iloc[train_indices]
        train_qualities = test_qualities.iloc[train_indices]
        test_queries = [test_queries[i] for i in test_indices]
        test_model_answers = test_model_answers.iloc[test_indices]
        test_costs = test_costs.iloc[test_indices]
        test_qualities = test_qualities.iloc[test_indices]
    else:
        train_model_answers = pd.read_json(os.path.join(data_folder, dataset, 'train', 'model_answers.json'))
        train_costs = pd.read_json(os.path.join(data_folder, dataset, 'train', 'costs.json'))
        train_qualities = pd.read_json(os.path.join(data_folder, dataset, 'train', 'qualities.json'))
        train_queries = pd.read_json(os.path.join(data_folder, dataset, 'train', 'queries.json'))
        train_queries = [(q[0], q[1]) for i, q in train_queries.iterrows()]
        

    # test_costs["20240620_sweagent_claude3.5sonnet"] *= 5 # for testing purposes
    # train_costs["20240620_sweagent_claude3.5sonnet"] *= 5 # for testing purposes
    # test_model_answers["20240620_sweagent_claude3.5sonnet"] = test_model_answers["20240620_sweagent_claude3.5sonnet"].apply(lambda x: x[:-1] + [x[-1] * 5]) # for testing purposes
    # train_model_answers["20240620_sweagent_claude3.5sonnet"] = train_model_answers["20240620_sweagent_claude3.5sonnet"].apply(lambda x: x[:-1] + [x[-1] * 5]) # for testing purposes

    test_qualities_averaged = test_qualities.mean(axis=0)
    test_costs_averaged = test_costs.mean(axis=0)
    train_qualities_averaged = train_qualities.mean(axis=0)
    train_costs_averaged = train_costs.mean(axis=0)

    print(test_costs_averaged)
    print(test_qualities_averaged)
    print(train_costs_averaged)
    print(train_qualities_averaged)

    models = [all_models[int(i)] for i in args.models.split(',')]
    models = sorted(models, key=lambda x: test_costs_averaged[x['name']])

    train_split = int(0.5 * len(train_queries))

    max_lambda = 1

    results = test_everything(models, n_iterations=10,
                            train_model_answers=train_model_answers,
                            train_costs=train_costs,
                            train_queries=train_queries,
                            test_model_answers=test_model_answers,
                            test_costs=test_costs,
                            test_queries=test_queries,
                            train_qualities=train_qualities,
                            test_qualities=test_qualities, 
                            test_costs_averaged=test_costs_averaged,
                            test_qualities_averaged=test_qualities_averaged,
                            dataset=dataset,
                            data_folder=data_folder,
                            max_lambda=max_lambda,
                            model_class = LogisticRegression,
                            n_cores=30, greedy=False, 
                            train_split=train_split, 
                            force_order=False, 
                            max_depth=None, 
                            n_samples=100, 
                            no_cascade_router=False, 
                            no_router=False, 
                            no_cascade=False, 
                            set_sigma_none=False, 
                            is_swebench=False,
                            is_code_math=True,
                            is_classification=False,
                            use_sum_costs=True,
                            cascade_strategies=[strat1, strat4], 
                            cascade_router_strategies=[strat1, strat4, strat3, strat2], 
                            is_latency_cost=False,
                            )

    filename = f'{args.models}.json'

    folder = f'data/results/{args.dataset}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(results, f)
