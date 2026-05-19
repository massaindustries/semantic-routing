from selection import APIQuery
import asyncio
import numpy as np
import pandas as pd
from datasets import load_dataset
import os
from loguru import logger


def compute_logprob(logprobs, answer):
    """
    Computes the log probability of an answer given a list of log probabilities.

    Args:
        logprobs (list): A list of log probabilities.
        answer (str): The answer for which the log probability is computed.

    Returns:
        float: The computed log probability.
    """
    logprob = 0
    total_now = ''
    for token_pos in logprobs[::-1]:
        token, logprob_token = token_pos[0]
        if total_now == '' and not answer.endswith(token) and not token.endswith(answer):
            continue
        total_now = token + total_now
        if logprob_token is not None:
            logprob += logprob_token
        if total_now.endswith(answer):
            break
    return logprob

def run_dataset(
        model,
        api,
        df,
        validation_df,
        num_fewshot,
        prompt_template_function,
        read_cost,
        write_cost,
        chat=False,
        sequential=False,
        **kwargs
):
    """
    Run the dataset through the model via an API and return the model answers, costs, and qualities.
    Args:
        model (str): The model to use.
        api (str): The API to use.
        df (pandas.DataFrame): The dataset to run.
        validation_df (pandas.DataFrame): The validation dataset.
        num_fewshot (int): The number of few-shot examples.
        prompt_template_function (function): The function to generate prompt templates.
        read_cost (float): The cost of reading from the API.
        write_cost (float): The cost of writing to the API.
        chat (bool, optional): Whether to use chat mode. Defaults to False.
        **kwargs: Additional keyword arguments.
    Returns:
        tuple: A tuple containing the output model answers, output costs, and output qualities.
    """
    queries = get_queries(df, validation_df, num_fewshot, prompt_template_function, chat)
    # queries = queries[-20:]

    api_query = APIQuery(
        model=model,
        temperature=0.0,
        max_tokens=1,
        max_retries=20,
        api=api,
        return_logprobs=True,
        logprobs=1,
        chat=chat,
        read_cost=read_cost,
        write_cost=write_cost,
        echo=True,
        requests_per_second=20 if api != "huggingface" else 1,
        sequential=sequential,
        **kwargs
    )

    outputs, detailed_cost, cost = asyncio.run(api_query.run_queries(queries))
    logger.info(f'Cost: {cost}')

    output_model_answers = []
    output_costs = []
    output_qualities = []

    current_index = 0
    
    for sample_index in range(len(df)):
        options = df['options'].iloc[sample_index]
        answer = df['answer'].iloc[sample_index]
        model_answers = []
        for i, option in enumerate(options):
            output = outputs[current_index]
            cost = detailed_cost[current_index]
            logprobs = compute_logprob(output[1], 'ABCDEFGHIJ'[i])
            model_answers.append(logprobs)
            current_index += 1
        
        model_answers = np.exp(model_answers)
        model_answers /= np.sum(model_answers)
        output_model_answers.append(list(model_answers))
        if sequential:
            output_costs.append(cost['time'])
        else:
            output_costs.append(cost['cost'] - cost['output_tokens'] / 10 ** 6 * write_cost) # because of the idiotic way in which the together api works, we have to do it this way
        output_qualities.append(float(options[np.argmax(model_answers)] == answer))
    return output_model_answers, output_costs, output_qualities

def get_queries(df, validation_df, num_fewshot, prompt_template_function, chat, include_output=True):
    """
    Generate queries based on the input dataframe and validation dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe.
        validation_df (pandas.DataFrame or str): The validation dataframe or the path to the validation dataframe.
        num_fewshot (int): The number of few-shot examples to include.
        prompt_template_function (function): The function to generate the prompt template.
        chat (bool): Whether to generate queries for chat or not.
        include_output (bool, optional): Whether to include output options in the queries. Defaults to True.

    Returns:
        list: The list of generated queries.
    """
    if num_fewshot > 0:
        if isinstance(validation_df, str):
            fewshot = validation_df
        else:
            fewshot_df = validation_df.sample(num_fewshot, random_state=0)
            fewshot = [prompt_template_function(input_, options, answer) for input_, options, answer in zip(fewshot_df['input'], 
                                                                                                            fewshot_df['options'], 
                                                                                                            fewshot_df['answer'])]
            if not chat:
                fewshot = '\n\n'.join(fewshot)
                fewshot += '\n\n'
    else:
        fewshot = ''
        if chat:
            fewshot = []

    queries = []
    for i in range(len(df)):
        input_ = df['input'].iloc[i]
        options = df['options'].iloc[i]
        if include_output:
            for option in options:
                query = fewshot + prompt_template_function(input_, options, option)
                queries.append(query)
        else:
            query = fewshot + prompt_template_function(input_, options, None)
            queries.append(query)
    return queries

def store_model_outputs(model_answers, costs, qualities, output_path):
    """
    Store the model outputs in a JSON file.

    Args:
        model_answers (list): List of model answers.
        costs (list): List of costs associated with each model answer.
        qualities (list): List of qualities associated with each model answer.
        output_path (str): Path to the output JSON file.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({
        'model_answers': model_answers,
        'costs': costs,
        'qualities': qualities
    })
    df.to_json(output_path, index=False)


def store_all_models(individual_model_paths, model_names, output_folder, queries):
    """
    Store the qualities, model answers, costs, and queries of multiple models in JSON format.

    Parameters:
    - individual_model_paths (list): List of file paths for individual model JSON files.
    - model_names (list): List of model names corresponding to the individual model paths.
    - output_folder (str): Path to the output folder where the JSON files will be stored.
    - queries (list): List of queries.

    Returns:
    None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    qualities = pd.DataFrame()
    model_answers = pd.DataFrame()
    costs = pd.DataFrame()
    for model_path, model_name in zip(individual_model_paths, model_names):
        if not os.path.isfile(model_path):
            continue
        df = pd.read_json(model_path)
        qualities[model_name] = df['qualities']
        model_answers[model_name] = df['model_answers']
        costs[model_name] = df['costs']
    qualities.to_json(f'{output_folder}/qualities.json', index=False)
    model_answers.to_json(f'{output_folder}/model_answers.json', index=False)
    costs.to_json(f'{output_folder}/costs.json', index=False)
    pd.DataFrame(queries).to_json(f'{output_folder}/queries.json', index=False)

def arc_challenge_prompt(input_, options, answer):
    """
    Generates a prompt for ARC challenge.

    Parameters:
    input_ (str): The question prompt.
    options (list): The list of answer options.
    answer (str): The correct answer.

    Returns:
    str: The generated prompt string.
    """
    output_string = f"Question: {input_}\n"
    correct_option = None
    for i, option in enumerate(options):
        string_option = 'ABCDEFGHIJ'[i]
        output_string += f"{string_option}: {option}\n"
        if option == answer:
            correct_option = string_option
    output_string += f"\nAnswer:"
    if answer is None:
        return output_string
    return output_string + f" {correct_option}"

def parse_arc_subset(subset):
    """
    Parses the ARC subset data.

    Args:
        subset (pandas.DataFrame): The subset data to be parsed.

    Returns:
        pandas.DataFrame: The parsed subset data with modified columns.
    """
    subset['input'] = subset['question']
    subset['options'] = subset['choices'].apply(lambda x: x['text'])
    translation = {str(i + 1): 'ABCDEFGHIJ'[i] for i in range(10)}
    subset['answer'] = subset.apply(lambda x: x['options']['ABCDEFGH'.index(translation.get(x['answerKey'], x['answerKey']))], axis=1)
    return subset

def parse_arc_challenge():
    """
    Parses the ARC-Challenge dataset from the 'allenai/ai2_arc' repository.

    Returns:
        train (DataFrame): The parsed training subset of the ARC-Challenge dataset.
        validation (DataFrame): The parsed validation subset of the ARC-Challenge dataset.
        test (DataFrame): The parsed test subset of the ARC-Challenge dataset.
    """
    dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge')
    train = parse_arc_subset(pd.DataFrame(dataset['train']))
    validation = parse_arc_subset(pd.DataFrame(dataset['validation']))
    test = parse_arc_subset(pd.DataFrame(dataset['test']))
    return train, validation, test

def mmlu_prompt(input_, options, answer):
    """
    Generates a prompt for MMLU multiple-choice questions.

    Parameters:
    input_ (str): The question prompt.
    options (list): A list of options for the question.
    answer (str): The correct answer for the question.

    Returns:
    str: The generated prompt string.
    """
    output_string = f"Question: {input_}\n"
    correct_option = None
    for i, option in enumerate(options):
        string_option = 'ABCDEFGHIJ'[i]
        output_string += f"{string_option}: {option}\n"
        if option == answer:
            correct_option = string_option
    output_string += f"\nAnswer:"
    if answer is None:
        return output_string
    return output_string + f" {correct_option}"

def parse_mmlu_subset(subset):
    """
    Parses the MMLU subset dataframe.

    Args:
        subset (pandas.DataFrame): The MMLU subset dataframe.

    Returns:
        pandas.DataFrame: The parsed MMLU subset dataframe with modified columns 'input' and 'answer'.
    """
    subset['input'] = subset['question']
    subset['answer'] = subset.apply(lambda x: x['options']['ABCDEFGHIJKLMNOP'.index(x['answer'])], axis=1)
    return subset

def parse_mmlu():
    """
    Parses the MMLU dataset and returns the train, validation, and test subsets.

    Returns:
        train (pandas.DataFrame): The training subset of the MMLU dataset.
        validation (pandas.DataFrame): The validation subset of the MMLU dataset.
        test (pandas.DataFrame): The test subset of the MMLU dataset.
    """
    dataset = load_dataset('TIGER-Lab/MMLU-Pro', 'default')
    train = parse_mmlu_subset(pd.DataFrame(dataset['test']))
    validation = parse_mmlu_subset(pd.DataFrame(dataset['validation']))
    train = train.sample(frac=1, random_state=0).reset_index(drop=True)
    test = train.iloc[1500:3000]
    train = train.iloc[:1500]
    return train, validation, test

def mixeval_prompt(input_, options, answer):
    """
    Generates a prompt for a MixEval multiple-choice question.

    Parameters:
    input_ (str): The input text for the prompt.
    options (list): A list of strings representing the answer options.
    answer (str): The correct answer option.

    Returns:
    str: The generated prompt string.
    """
    output_string = f"{input_}\n"
    correct_option = None
    for i, option in enumerate(options):
        string_option = 'ABCDEFGHIJ'[i]
        output_string += f"{string_option}. {option}\n"
        if option == answer:
            correct_option = string_option
    output_string += f"Answer with the option letter from the given choices directly.\n"
    if answer is None:
        return output_string
    return output_string + f"{correct_option}"

def parse_mixeval(num_fewshot=3):
    """
    Parse the MixEval dataset for classification inference.

    Args:
        num_fewshot (int): Number of few-shot examples to include in the prefix.

    Returns:
        tuple: A tuple containing the train data, the prefix for few-shot examples, and the test data.
    """
    dataset = pd.DataFrame(load_dataset('MixEval/MixEval', 'MixEval')['multiple_choice'])
    dataset['input'] = dataset.apply(lambda x: x['prompt'] if x['context'] is None else x['context'] + '\n' + x['prompt'], axis=1)
    dataset['answer'] = dataset.apply(lambda x: x['options']['0123456789'.index(x['target'][0])], axis=1)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    train_data = dataset.iloc[:len(dataset) // 2].reset_index(drop=True)
    test_data = dataset.iloc[len(dataset) // 2:].reset_index(drop=True)
    FIVE_SHOT_PREFIX_MULTIPLECHOICE = \
    '''According to cell classification, prokaryotic cells are separated from eukaryotic cells. Which feature is often used to distinguish prokaryotic cells from eukaryotic cells?
A. life processes
B. size differences
C. plasma membranes
D. energy molecules
Answer with the option letter from the given choices directly.
B

As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective.
is elder scrolls online the same as skyrim
A. No
B. Yes
Answer with the option letter from the given choices directly.
A

connection
You can share files with someone if you have a connection to a what?
A. freeway
B. radio
C. wires
D. computer network
E. electrical circuit
Answer with the option letter from the given choices directly.
D

Approximately what percentage of participants in Milgram's obedience experiments thought they delivered the maximum amount of shock possible?
A. 0
B. 20
C. 40
D. 60
Answer with the option letter from the given choices directly.
D

How to check your Facebook feed
Which solution is correct?
A. Log in to Facebook. Click on the bell shaped button at the top right of your Facebook home window.
B. Log in to Facebook. Click on the bell shaped button at the top left of your Facebook home window.
Answer with the option letter from the given choices directly.
A'''
    FIVE_SHOT_PREFIX_MULTIPLECHOICE = '\n\n'.join(FIVE_SHOT_PREFIX_MULTIPLECHOICE.split('\n\n')[:num_fewshot]) + '\n\n'
    return train_data, FIVE_SHOT_PREFIX_MULTIPLECHOICE, test_data

def main(models, dataset, output_folder, num_fewshot=3, api='together', max_samples=None, 
         sequential=False):
    """
    Run classification inference on the specified dataset using the given models.
    Args:
        models (list): A list of dictionaries representing the models to be used. Each dictionary should have a 'name' key specifying the model name, and 'read_cost' and 'write_cost' keys specifying the costs associated with reading and writing.
        dataset (str): The dataset to be used for classification inference. Supported options are 'arc', 'mixeval', and 'mmlu'.
        output_folder (str): The folder where the output files will be stored.
        num_fewshot (int, optional): The number of few-shot examples to be used. Defaults to 3.
        api (str, optional): The API to be used for classification inference. Defaults to 'together'.
        max_samples (int, optional): The maximum number of samples to be used from the training and test datasets. Defaults to None.
    Raises:
        ValueError: If the specified dataset is not supported.
    Returns:
        None
    """
    if dataset == 'arc':
        train, validation, test = parse_arc_challenge()
        prompt = arc_challenge_prompt
    elif dataset == 'mixeval':
        train, validation, test = parse_mixeval(num_fewshot)
        prompt = mixeval_prompt
    elif dataset == 'mmlu':
        train, validation, test = parse_mmlu()
        prompt = mmlu_prompt
    else:
        raise ValueError(f'Dataset {dataset} not supported')
    
    if max_samples is not None:
        train = train.iloc[:max_samples]
        test = test.iloc[:max_samples]

    for df_name, df in zip(['train', 'test'], [train, test]):
        queries = get_queries(df, validation, num_fewshot, prompt, chat=False, include_output=False)
        for model in models:
            if not os.path.isfile(f'{output_folder}/{df_name}/{model["name"]}.json'):
                api_here = api if not model.get('is_huggingface', False) else 'huggingface'
                model_answers, costs, qualities = run_dataset(
                    model=model['name'],
                    api=api_here,
                    df=df,
                    validation_df=validation,
                    num_fewshot=num_fewshot,
                    prompt_template_function=prompt,
                    read_cost=model['read_cost'],
                    write_cost=model['write_cost'],
                    chat=False,
                    sequential=sequential
                )
                store_model_outputs(model_answers, costs, qualities, f'{output_folder}/{df_name}/{model["name"]}.json')
        store_all_models([f'{output_folder}/{df_name}/{model["name"]}.json' for model in models], [model['name'] for model in models], f"{output_folder}/{df_name}", 
                         queries=queries)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='arc')
    parser.add_argument('--output_folder', type=str, default='data/classification')
    parser.add_argument('--num_fewshot', type=int, default=1)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--sequential', action='store_true')

    args = parser.parse_args()

    models = [
        {
            'name': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
            'read_cost': 0.18,
            'write_cost': 0.18
        },
        {
            'name': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            'read_cost': 0.88,
            'write_cost': 0.88
        },
        {
            'name': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
            'read_cost': 5,
            'write_cost': 5
        },
        {
            'name': 'google/gemma-2-9b-it',
            'read_cost': 0.3,
            'write_cost': 0.3
        },
        {
            'name': 'google/gemma-2-27b-it',
            'read_cost': 0.8,
            'write_cost': 0.8
        },
        {
            'name': 'google/gemma-2b-it',
            'read_cost': 0.1,
            'write_cost': 0.1
        },
        {
            'name': 'mistralai/Mistral-7B-Instruct-v0.3',
            'read_cost': 0.2,
            'write_cost': 0.2
        },
        {
            'name': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
            'read_cost': 1.2,
            'write_cost': 1.2
        },
        {
            'name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'read_cost': 0.6,
            'write_cost': 0.6
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
            'read_cost': 0.3,
            'write_cost': 0.3
        },
        {
            'name': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
            'read_cost': 1.2,
            'write_cost': 1.2
        },
        {
            'name': 'microsoft/Phi-3.5-mini-instruct',
            'read_cost': 0.2,
            'write_cost': 0.2,
            "is_huggingface": True
        },
        # {
        #     'name': 'microsoft/Phi-3-small-8k-instruct',
        #     'read_cost': 0.3,
        #     'write_cost': 0.3,
        #     "is_huggingface": True
        # },
        # {
        #     'name': 'microsoft/Phi-3-medium-128k-instruct',
        #     'read_cost': 0.5,
        #     'write_cost': 0.5,
        #     "is_huggingface": True
        # },
    ]

    output_folder = f"{args.output_folder}/{args.dataset}"
    if args.sequential:
        output_folder += '_sequential'
    main(models, args.dataset, output_folder, num_fewshot=args.num_fewshot, api='together', 
         max_samples=args.samples, sequential=args.sequential)

