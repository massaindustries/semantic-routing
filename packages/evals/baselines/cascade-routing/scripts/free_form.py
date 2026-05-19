import pandas as pd
from selection import test_everything, ConstantStrategy, RepetitiveConstantStrategy, HyperoptStrategy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from tqdm import tqdm
from scipy.interpolate import interp1d
from transformers import AutoTokenizer
from concurrent.futures import as_completed, ProcessPoolExecutor
from loguru import logger
import sys
import os
import json
from collections import Counter
from copy import deepcopy
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
            'name': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
            'huggingface_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'read_cost': 0.18 * 10 ** -6,
            'write_cost': 0.18 * 10 ** -6
        },
        {
            'name': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            'huggingface_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'read_cost': 0.88 * 10 ** -6,
            'write_cost': 0.88 * 10 ** -6
        },
        {
            'name': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
            'huggingface_name': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
            'read_cost': 5 * 10 ** -6,
            'write_cost': 5 * 10 ** -6
        },
        {
            'name': 'google/gemma-2-9b-it',
            'read_cost': 0.3 * 10 ** -6,
            'write_cost': 0.3 * 10 ** -6
        },
        {
            'name': 'google/gemma-2-27b-it',
            'read_cost': 0.8 * 10 ** -6,
            'write_cost': 0.8 * 10 ** -6
        },
        {
            'name': 'google/gemma-2b-it',
            'read_cost': 0.1 * 10 ** -6,
            'write_cost': 0.1 * 10 ** -6
        },
        {
            'name': 'mistralai/Mistral-7B-Instruct-v0.3',
            'read_cost': 0.2 * 10 ** -6,
            'write_cost': 0.2 * 10 ** -6
        },
        {
            'name': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
            'read_cost': 1.2 * 10 ** -6,
            'write_cost': 1.2 * 10 ** -6
        },
        {
            'name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'read_cost': 0.6 * 10 ** -6,
            'write_cost': 0.6 * 10 ** -6
        },
        {
            'name': 'Qwen/Qwen2.5-0.5B-Instruct',
            'read_cost': 0.03,
            'write_cost': 0.03,
            "is_huggingface": True
        },
        {
            'name': 'Qwen/Qwen2.5-1.5B-Instruct',
            'read_cost': 0.1,
            'write_cost': 0.1,
            "is_huggingface": True
        },
        {
            'name': 'Qwen/Qwen2.5-3B-Instruct',
            'read_cost': 0.2,
            'write_cost': 0.2,
            "is_huggingface": True
        },
        {
            'name': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
            'huggingface_name': 'Qwen/Qwen2.5-7B-Instruct',
            'read_cost': 0.3,
            'write_cost': 0.3
        },
        {
            'name': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
            'huggingface_name': 'Qwen/Qwen2.5-72B-Instruct',
            'read_cost': 1.2,
            'write_cost': 1.2
        },
        {
            'name': 'microsoft/Phi-3.5-mini-instruct',
            'read_cost': 0.2,
            'write_cost': 0.2,
            "is_huggingface": True
        },
    ]

def linear_model():
    return LogisticRegression(max_iter=1000)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run classification experiments")
    parser.add_argument("--dataset", type=str, default="mmlu_gsm8k", help="Dataset name")
    parser.add_argument('--models', type=str, default='0,1,2')
    parser.add_argument('--assume-constant-costs', action='store_true')

    args = parser.parse_args()

    dataset = args.dataset
    data_folder = 'data/free_form'

    train_model_answers = pd.read_json(os.path.join(data_folder, dataset, 'train', 'model_answers.json'))
    train_costs = pd.read_json(os.path.join(data_folder, dataset, 'train', 'costs.json'))
    train_qualities = pd.read_json(os.path.join(data_folder, dataset, 'train', 'qualities.json'))
    train_queries = pd.read_json(os.path.join(data_folder, dataset, 'train', 'queries.json'))
    test_model_answers = pd.read_json(os.path.join(data_folder, dataset, 'test', 'model_answers.json'))
    test_costs = pd.read_json(os.path.join(data_folder, dataset, 'test', 'costs.json'))
    test_qualities = pd.read_json(os.path.join(data_folder, dataset, 'test', 'qualities.json'))
    test_queries = pd.read_json(os.path.join(data_folder, dataset, 'test', 'queries.json'))

    train_queries = list(train_queries.apply(lambda x: [x[0][0][1] + '\nQuestion: ' + x[1][1] + '\nAnswer:' + x[2][1] + '\nQuestion:' + x[3][1]] + x[0][1:], axis=1))
    test_queries = list(test_queries.apply(lambda x: [x[0][0][1] + '\nQuestion: ' + x[1][1] + '\nAnswer:' + x[2][1] + '\nQuestion:' + x[3][1]] + x[0][1:], axis=1))

    test_qualities_averaged = test_qualities.mean(axis=0)
    test_costs_averaged = test_costs.mean(axis=0)

    train_qualities_averaged = train_qualities.mean(axis=0)
    train_costs_averaged = train_costs.mean(axis=0)

    models = [all_models[int(i)] for i in args.models.split(',')]

    models = sorted(models, key=lambda x: train_costs_averaged[x['name']])

    train_split = int(0.5 * len(train_queries))

    max_lambda = 10000
    if args.assume_constant_costs:
        max_lambda = 5

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
                        model_class = linear_model,
                        n_cores=30, greedy=False, train_split=train_split,
                        force_order=True, max_depth=4, n_samples=100, 
                        no_cascade_router=False, no_router=False, no_cascade=False, 
                        set_sigma_none=False, is_classification=False,
                        cascade_strategies=[strat1, strat4], 
                        cascade_router_strategies=[strat1, strat4, strat3, strat2], 
                        is_latency_cost=args.assume_constant_costs)

    filename = f'{args.models}.json'

    folder = f'data/results/free_form/{args.dataset}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, filename), 'w') as f:
        json.dump(results, f)