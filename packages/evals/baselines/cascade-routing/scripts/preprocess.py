import pandas as pd
import numpy as np
import os
import pickle
import pandas as pd

def pkl2csv(pkl_path, save_csv_path):
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    df = pd.DataFrame(data)
    df.to_csv(save_csv_path, index=False)

def combine_datasets(dataset_names, train_test=True):
    storage_folder = '_'.join(dataset_names)
    if train_test:
        os.makedirs(os.path.join(data_folder, storage_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, storage_folder, 'test'), exist_ok=True)
    if train_test:
        train_model_answers = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'train', 'model_answers.json')) for dataset in dataset_names])
        train_costs = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'train', 'costs.json')) for dataset in dataset_names])
        train_qualities = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'train', 'qualities.json')) for dataset in dataset_names])
        train_queries = []
    test_queries = []
    for i, dataset in enumerate(dataset_names):
        additional_features = np.zeros(len(dataset_names))
        additional_features[i] = 1
        additional_features = list(additional_features)
        if train_test:
            new_train = pd.read_json(os.path.join(data_folder, dataset, 'train', 'queries.json'))
            new_train[0] = new_train[0].apply(lambda x: [x] + additional_features)
            train_queries.append(new_train)
        if train_test:
            new_test = pd.read_json(os.path.join(data_folder, dataset, 'test', 'queries.json'))
        else:
            new_test = pd.read_json(os.path.join(data_folder, dataset, 'queries.json'))
        new_test[0] = new_test[0].apply(lambda x: [x] + additional_features)
        test_queries.append(new_test)

    test_queries = pd.concat(test_queries)
    if train_test:
        train_queries = pd.concat(train_queries)
       
        test_model_answers = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'test', 'model_answers.json')) for dataset in dataset_names])
        test_costs = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'test', 'costs.json')) for dataset in dataset_names])
        test_qualities = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'test', 'qualities.json')) for dataset in dataset_names])
    else:
        test_model_answers = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'model_answers.json')) for dataset in dataset_names])
        test_costs = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'costs.json')) for dataset in dataset_names])
        test_qualities = pd.concat([pd.read_json(os.path.join(data_folder, dataset, 'qualities.json')) for dataset in dataset_names])

    # shuffle
    np.random.seed(42)
    if train_test:
        train_indices = np.random.permutation(len(train_model_answers))
        train_model_answers = train_model_answers.iloc[train_indices].reset_index(drop=True)
        train_costs = train_costs.iloc[train_indices].reset_index(drop=True)
        train_qualities = train_qualities.iloc[train_indices].reset_index(drop=True)
        train_queries = train_queries.iloc[train_indices].reset_index(drop=True)
    test_indices = np.random.permutation(len(test_model_answers))
    test_model_answers = test_model_answers.iloc[test_indices].reset_index(drop=True)
    test_costs = test_costs.iloc[test_indices].reset_index(drop=True)
    test_qualities = test_qualities.iloc[test_indices].reset_index(drop=True)
    test_queries = test_queries.iloc[test_indices].reset_index(drop=True)
    # save
    if train_test:
        train_model_answers.to_json(os.path.join(data_folder, storage_folder, 'train', 'model_answers.json'), index=False)
        train_costs.to_json(os.path.join(data_folder, storage_folder, 'train', 'costs.json'), index=False)
        train_qualities.to_json(os.path.join(data_folder, storage_folder, 'train', 'qualities.json'), index=False)
        train_queries.to_json(os.path.join(data_folder, storage_folder, 'train', 'queries.json'), index=False)
    test_model_answers.to_json(os.path.join(data_folder, storage_folder, 'test', 'model_answers.json'), index=False)
    test_costs.to_json(os.path.join(data_folder, storage_folder, 'test', 'costs.json'), index=False)
    test_qualities.to_json(os.path.join(data_folder, storage_folder, 'test', 'qualities.json'), index=False)
    test_queries.to_json(os.path.join(data_folder, storage_folder, 'test', 'queries.json'), index=False)

if __name__ == '__main__':
    pkl2csv('../data/routerbench_0shot.pkl', '../data/routerbench_0shot.csv')
    pkl2csv('../data/routerbench_5shot.pkl', '../data/routerbench_5shot.csv')
    pkl2csv('../data/routerbench_raw.pkl', '../data/routerbench_raw.csv')
    data_folder = 'data/code_math'

    combine_datasets(['livecodebench', 'minerva'], False)
    data_folder = 'data/classification'

    combine_datasets(['mmlu', 'arc', 'mixeval'])
    combine_datasets(['mmlu_sequential', 'arc_sequential', 'mixeval_sequential'])

    data_folder = 'data/free_form'

    combine_datasets(['mmlu', 'gsm8k'])
    combine_datasets(['mmlu_sequential', 'gsm8k_sequential'])