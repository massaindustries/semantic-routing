from selection import test_everything
from selection import HyperoptStrategy, ConstantStrategy
from loguru import logger
import sys
import os
import pandas as pd
import numpy as np
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

def strat2(max_lambda):
    return HyperoptStrategy(max_lambda, 200, max_factor=4)

def strat3(max_lambda):
    return HyperoptStrategy(max_lambda, 200, max_factor=4, from_scratch=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run routerbench')
    parser.add_argument('--models', type=str, default='0,1,2')
    parser.add_argument('--noise-level', type=str, default='low')
    parser.add_argument('--few-shot', action='store_true')
    parser.add_argument("--noise-before", type=float, default=None)
    parser.add_argument("--noise-after", type=float, default=None)
    parser.add_argument("--cost-noise-before", type=float, default=None)
    parser.add_argument("--cost-noise-after", type=float, default=None)
    parser.add_argument("--extensive-file", action='store_true')

    args = parser.parse_args()

    if args.few_shot:
        data = pd.read_csv('data/routerbench_5shot.csv')
        # fillna with 0
        data = data.fillna(0)
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
        strat1, strat2, strat3
    ]

    cascade_router_strategies = [
        strat1, strat2, strat3
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
    
    if args.noise_before is not None:
        ground_truth_noise_before = args.noise_before
    if args.noise_after is not None:
        ground_truth_noise_after = args.noise_after
    if args.cost_noise_before is not None:
        cost_noise_before = args.cost_noise_before
    if args.cost_noise_after is not None:
        cost_noise_after = args.cost_noise_after

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
                        greedy=False, 
                        train_split=0, 
                        force_order=False, 
                        max_depth=5, 
                        n_samples=100, 
                        ground_truth_noise_before=ground_truth_noise_before, 
                        ground_truth_noise_after=ground_truth_noise_after, 
                        cost_noise_before=cost_noise_before,
                        cost_noise_after=cost_noise_after,
                        ground_truth_cost_computer=True, 
                        set_sigma_none=False, 
                        is_routerbench=True, 
                        cascade_strategies=cascade_strategies,
                        cascade_router_strategies=cascade_router_strategies,
    )
    filename = f'{args.models}_{args.noise_level}_{"5shot" if args.few_shot else "0shot"}'
    if args.extensive_file:
        str_noises = f"{ground_truth_noise_before},{ground_truth_noise_after},{cost_noise_before},{cost_noise_after}_False"
        filename += f'_{str_noises}.json'
    else:
        filename += '.json'

    folder = 'data/results/routerbench'
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(results, f)