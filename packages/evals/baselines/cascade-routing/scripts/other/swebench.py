from datasets import load_dataset
import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np

def import_full_swebench():
    swebench = None
    for split in ['_Verified', '_Lite', '']:
        swebench_split = load_dataset(f'princeton-nlp/SWE-bench{split}', split="test").to_pandas()
        swebench_split['split'] = split
        # remove all samples for which "instance_id" is already in swebench
        swebench_split = swebench_split[~swebench_split['instance_id'].isin(swebench['instance_id'])] if swebench is not None else swebench_split
        if swebench is None:
            swebench = swebench_split
        else:
            swebench = pd.concat([swebench, swebench_split])
    return swebench

group_parser = {
    "verified": "_Verified",
    "lite": "_Lite",
    "test": ""
}

def parse_trajectory(group, model_name, problem_id, swebench):
    path = f"evaluation/{group}/{model_name}/trajs/{problem_id}.traj"
    problem_statement = swebench[np.logical_and(swebench['instance_id'] == problem_id, swebench["split"] == group_parser[group])]['problem_statement'].values[0]
    repo = problem_id.split('-')[0]
    trajectory = json.load(open(path, 'r'))
    cost = trajectory['info']['model_stats']['total_cost']
    resolved_model_problem_ids = json.load(open(f"evaluation/{group}/{model_name}/results/results.json"))['resolved']
    solved = int(problem_id in resolved_model_problem_ids)
    return {
        "query": [repo, problem_statement],
        "cost": cost,
        "solved": solved,
        "model_answer": [trajectory["info"].get('exit_status', "None"), solved, trajectory["info"]["model_stats"]['api_calls'], trajectory["info"]["model_stats"]['tokens_sent'], cost],
        "problem_id": problem_id
    }

def parse_model(group, model_name, swebench):
    results = []
    try:
        all_problem_ids = os.listdir(f"evaluation/{group}/{model_name}/trajs")
        # remove extension
        all_problem_ids = [problem_id.split('.')[0] for problem_id in all_problem_ids]
    except:
        return None
    # sort alphabetically
    all_problem_ids.sort()
    n_problems = 0
    problems = []
    for problem_id in tqdm(all_problem_ids):
        try:
            results.append(parse_trajectory(group, model_name, problem_id, swebench))
        except Exception as e:
            n_problems += 1
            problems.append((problem_id, str(e)))
            continue
    print(f"Problems encountered: {n_problems} / {len(all_problem_ids)}")
    print('Example problems:')
    for problem in problems[:5]:
        print(problem)
    if len(results) <= 100: # require minimum number to have been parsed
        return None
    return results

def parse_group(group, swebench):
    models = os.listdir(f"evaluation/{group}")
    all_results = dict()
    for model in models:
        if "sweagent" in model:
            results = parse_model(group, model, swebench)
            if results is not None:
                all_results[model] = results
    return remove_not_present_all(all_results)

def remove_not_present_all(all_results):
    intersection_problems = set.intersection(*[set([x['problem_id'] for x in v]) for v in all_results.values()])
    intersection_problems = list(intersection_problems)
    # sort everything by the order in intersection_problems
    all_results_new = dict()
    for model_name in all_results:
        all_results_new[model_name] = [x for x in all_results[model_name] if x['problem_id'] in intersection_problems]
        all_results_new[model_name].sort(key=lambda x: intersection_problems.index(x['problem_id']))
    return all_results_new

def create_appropriate_data(data):
    qualities = pd.DataFrame()
    model_answers = pd.DataFrame()
    costs = pd.DataFrame()
    queries = []

    for model_name in data:
        qualities[model_name] = [data[model_name][i]['solved'] for i in range(len(data[model_name]))]
        model_answers[model_name] = [data[model_name][i]['model_answer'] for i in range(len(data[model_name]))]
        costs[model_name] = [data[model_name][i]['cost'] for i in range(len(data[model_name]))]
        queries = [data[model_name][i]['query'] for i in range(len(data[model_name]))]
    
    return qualities, model_answers, costs, pd.DataFrame(queries)


swebench = import_full_swebench()
verified_data = parse_group('verified', swebench)
validation_data = parse_group('test', swebench)

qualities, model_answers, costs, queries = create_appropriate_data(verified_data)

os.makedirs("data/test", exist_ok=True)

qualities.to_json("data/test/qualities.json")
model_answers.to_json("data/test/model_answers.json")
costs.to_json("data/test/costs.json")
queries.to_json("data/test/queries.json")

qualities, model_answers, costs, queries = create_appropriate_data(validation_data)

os.makedirs("data/train", exist_ok=True)
qualities.to_json("data/train/qualities.json")
model_answers.to_json("data/train/model_answers.json")
costs.to_json("data/train/costs.json")
queries.to_json("data/train/queries.json")
