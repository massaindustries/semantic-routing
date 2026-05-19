from selection import test_everything, remove_redundant_models
from selection import HyperoptStrategy, ConstantStrategy
from loguru import logger
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import numpy as np

np.random.seed(0)

# set logger to only show info messages
logger.remove()
logger.add(sys.stdout, level="INFO")

def strat1(max_lambda):
    return ConstantStrategy(max_lambda, n_iterations=30)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run routerbench')
    parser.add_argument('--models', type=str, default='0,1,2')
    parser.add_argument('--noise-level', type=str, default='low')
    parser.add_argument('--few-shot', action='store_true')
    parser.add_argument('--no-speedup', action='store_true')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--sigma-none', action='store_true')

    args = parser.parse_args()

    if args.few_shot:
        data = pd.read_csv('data/routerbench_5shot.csv')
    else:
        data = pd.read_csv('data/routerbench_0shot.csv')

    model_names = data.columns[3:14]

    qualities = data[model_names]
    cost_names = [model_name + '|total_cost' for model_name in model_names]
    costs = data[cost_names]
    # change column names
    costs.columns = model_names
    answer_names = [model_name + '|model_response' for model_name in model_names]
    answers = data[answer_names]
    # change column names
    answers.columns = model_names

    queries = np.array(data[['prompt']]).reshape(-1)

    qualities_train, qualities_test, costs_train, costs_test, answers_train, answers_test, queries_train, queries_test = train_test_split(
                                qualities, costs, answers, queries, test_size=0.95, random_state=42                            
    )

    train_qualities_averaged = qualities_train.mean(axis=0)
    train_costs_averaged = costs_train.mean(axis=0)

    test_qualities_averaged = qualities_test.mean(axis=0)
    test_costs_averaged = costs_test.mean(axis=0)

    model_names_dict = [{'name': model_name} for model_name in model_names]

    models = [model_names_dict[int(i)] for i in args.models.split(',')]

    # sort models by cost, cheapest first
    models = sorted(models, key=lambda x: train_costs_averaged[x['name']])

    cascade_strategies = [
        strat1
    ]

    cascade_router_strategies = [
        strat1
    ]

    if args.noise_level == 'low':
        ground_truth_noise_before=0.6
        ground_truth_noise_after=0.3
        cost_noise_before=0.0002
        cost_noise_after=0.00005
    elif args.noise_level == 'medium':
        ground_truth_noise_before=1.6
        ground_truth_noise_after=0.8 
        cost_noise_before=0.0004
        cost_noise_after=0.0001
    else:
        ground_truth_noise_before=2.4
        ground_truth_noise_after=1.2 
        cost_noise_before=100
        cost_noise_after=100

    results = test_everything(models,
                        train_model_answers=answers_train,
                        train_costs=costs_train,
                        train_queries=queries_train,
                        test_model_answers=answers_test,
                        test_costs=costs_test,
                        test_queries=queries_test,
                        train_qualities=qualities_train,
                        test_qualities=qualities_test, 
                        test_costs_averaged=test_costs_averaged,
                        test_qualities_averaged=test_qualities_averaged,
                        dataset=None,
                        data_folder=None,
                        n_iterations=10, 
                        max_lambda=10000,
                        model_class=LogisticRegression,
                        n_cores=50, 
                        greedy=args.greedy, 
                        train_split=0, 
                        force_order=True, 
                        max_depth=5, 
                        n_samples=100, 
                        ground_truth_noise_before=ground_truth_noise_before, 
                        ground_truth_noise_after=ground_truth_noise_after, 
                        cost_noise_before=cost_noise_before,
                        cost_noise_after=cost_noise_after,
                        ground_truth_cost_computer=True, 
                        set_sigma_none=args.sigma_none, 
                        is_routerbench=True, 
                        no_router=True,
                        no_cascade=True,
                        cascade_strategies=cascade_strategies,
                        cascade_router_strategies=cascade_router_strategies,
                        do_speedup=not args.no_speedup
    )

    filename = f'{args.models}_{args.noise_level}_{"5shot" if args.few_shot else "0shot"}_{args.no_speedup}_{args.greedy}_{args.sigma_none}.json'

    folder = 'data/results/routerbench_times'
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(results, f)