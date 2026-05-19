from selection import APIQuery
import asyncio
import numpy as np
import pandas as pd
from datasets import load_dataset
import os
from loguru import logger
from classification_inference import store_all_models, store_model_outputs
import re

def run_dataset(
        model,
        api,
        df,
        validation_df,
        num_fewshot,
        system_message,
        read_cost,
        write_cost,
        parse_answer_function,
        sequential=False,
        **kwargs
):
    """
    Run the dataset through the model and return the model answers, costs, and qualities.
    Args:
        model (str): The model to use.
        api (str): The API to use.
        df (pandas.DataFrame): The dataset to run.
        validation_df (pandas.DataFrame): The validation dataset.
        num_fewshot (int): The number of few-shot examples.
        system_message (str): The system message.
        read_cost (float): The cost of reading.
        write_cost (float): The cost of writing.
        parse_answer_function (function): The function to parse the answer.
        **kwargs: Additional keyword arguments.
    Returns:
        tuple: A tuple containing the model answers, costs, and qualities.
            - output_model_answers (list): A list of tuples containing the model answer, logprobs, and parsed answer.
            - output_costs (list): A list of costs for each query.
            - output_qualities (list): A list of qualities for each query.
    """
    queries = get_queries(df, validation_df, num_fewshot, system_message)

    api_query = APIQuery(
        model=model,
        temperature=0.7,
        max_tokens=1024,
        max_retries=20,
        timeout=120,
        api=api,
        return_logprobs=True,
        logprobs=1,
        chat=True,
        read_cost=read_cost,
        write_cost=write_cost,
        requests_per_second=5 if api != "huggingface" else 1,
        sequential=sequential,
        **kwargs
    )

    outputs, detailed_cost, cost = asyncio.run(api_query.run_queries(queries))

    logger.info(f'Cost: {cost}')

    output_model_answers = []
    output_costs = []
    output_qualities = []

    for i, output in enumerate(outputs):
        parsed_answer = parse_answer_function(output[0])
        if parsed_answer is None:
            if 'options' in df.columns:
                parsed_answer = np.random.choice(list('ABCDEFGHIJ'[:len(df.iloc[i]['options'])]))
            else:
                parsed_answer = 0
        output_model_answers.append((output[0], [token[0][1] for token in output[1] if token is not None], parsed_answer))
        if sequential:
            output_costs.append(detailed_cost[i]['time'])
        else:
            output_costs.append(detailed_cost[i]['cost'])
        
        correct = float(parsed_answer == df.iloc[i]['answer'])
        output_qualities.append(correct)

    return output_model_answers, output_costs, output_qualities

def get_queries(df, validation_df, num_fewshot, system_message):
    """
    Generate a list of queries for the dataset.

    Args:
        df (pandas.DataFrame): The input dataframe.
        validation_df (pandas.DataFrame): The validation dataframe.
        num_fewshot (int): The number of few-shot examples to include.
        system_message (str): The system message to include in the queries.

    Returns:
        list: A list of queries, where each query is a list of tuples representing the message source and content.
    """

    if num_fewshot > 0:
        fewshot_df = validation_df.sample(num_fewshot, random_state=0)
        fewshot = []
        for i, row in fewshot_df.iterrows():
            fewshot.append(('human', row['input']))
            fewshot.append(('ai', row['output']))
    else:
        fewshot = []

    queries = []
    for i, row in df.iterrows():
        query = [('system', system_message)] + fewshot + [
            ('human', row['input'])
        ]
        queries.append(query)
    return queries

def mmlu_prompt(input_, options):
    """
    Generates a prompt string for the MMLU multiple choice selection.

    Parameters:
    input_ (str): The input string to be displayed as the prompt.
    options (list): A list of options to be displayed as choices.

    Returns:
    str: The generated prompt string.
    """
    output_string = f"{input_}\n"
    for i, option in enumerate(options):
        string_option = 'ABCDEFGHIJ'[i]
        output_string += f"{string_option}: {option}\n"
    return output_string

def parse_mmlu_subset(subset):
    """
    Parses the given subset of data for MMLU (Mean Length of Utterance) inference.

    Args:
        subset (pandas.DataFrame): The subset of data to be parsed.

    Returns:
        pandas.DataFrame: The parsed subset with modified 'input' and 'output' columns.
    """
    subset['input'] = subset['question']
    subset['input'] = subset.apply(lambda x: mmlu_prompt(x['input'], x['options']), axis=1)
    subset['output'] = subset['cot_content'].apply(lambda x: x[3:]) # remove the "A: " prefix
    return subset

def mmlu_parser(answer):
    """
    Parses the answer string and extracts the answer.

    Args:
        answer (str): The answer string to be parsed.

    Returns:
        str or None: The extracted answer if found, otherwise None.
    """
    regex1 = re.compile(r"answer is \(?\([A-J]\)?\)", re.IGNORECASE)
    regex_match = regex1.search(answer)
    if regex_match is None:
        regex2 = re.compile(r"\.*[aA]nswer:\s*\([A-J]\)")
        regex_match = regex2.search(answer)
        if regex_match is None:
            # answer is without brackets
            regex3 = re.compile(r"answer is [A-J]", re.IGNORECASE)
            regex_match = regex3.search(answer)
            if regex_match is None:
                return None
            else:
                return regex_match.group()[-1]
        else:
            return regex_match.group()[-2]
    else:
        return regex_match.group()[-2]
    # get the last group of the match and return that answer

def parse_mmlu():
    """
    Parses the MMLU dataset and returns the train, validation, and test sets along with a system prompt.

    Returns:
        train (pandas.DataFrame): The training set.
        validation (pandas.DataFrame): The validation set.
        test (pandas.DataFrame): The test set.
        system_prompt (str): The system prompt for multiple choice questions.
    """
    # code implementation
    dataset = load_dataset('TIGER-Lab/MMLU-Pro', 'default')
    train = parse_mmlu_subset(pd.DataFrame(dataset['test']))
    validation = parse_mmlu_subset(pd.DataFrame(dataset['validation']))
    train = train.sample(frac=1, random_state=0).reset_index(drop=True)
    test = train.iloc[1500:3000]
    train = train.iloc[:1500]
    system_prompt = "The following are multiple choice questions (with answers). Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice."
    return train, validation, test, system_prompt

def parse_gsm8k_subset(subset):
    """
    Parse the GSM8K subset by assigning the 'question' column to the 'input' column,
    the 'answer' column to the 'output' column, and applying the 'gsm8k_parser' function
    to the 'output' column.

    Parameters:
    subset (pandas.DataFrame): The subset of GSM8K data to be parsed.

    Returns:
    pandas.DataFrame: The parsed subset with the 'input', 'output', and 'answer' columns updated.
    """
    subset['input'] = subset['question']
    subset['output'] = subset['answer']
    subset['answer'] = subset['output'].apply(gsm8k_parser)
    return subset

def gsm8k_parser(answer):
    """
    Parses the answer string and extracts the integer answer

    Args:
        answer (str): The answer string to be parsed.

    Returns:
        int or None: The last group of digits as an integer if found, None otherwise.
    """
    # compile first to match last "#### NUMBER"
    try:
        answer = answer.replace('.', '').replace(',', '') # answers can only be integers
        regex1 = re.compile(r"####\s*[0-9]+")
        regex_match = regex1.search(answer)
        if regex_match is None:
            regex2 = re.compile(r"###\s*[0-9]+")
            regex_match = regex2.search(answer)
            if regex_match is None:
                # get the last group of digits and return that
                regex3 = re.compile(r"[0-9]+")
                # find all the numbers in the string
                all_results = regex3.findall(answer)
                if len(all_results) == 0:
                    return None
                return int(all_results[-1])
            else:
                return int(regex_match.group()[4:].strip())
        return int(regex_match.group()[5:].strip())
    except:
        return None


def parse_gsm8k():
    """
    Parses the GSM8K dataset and returns the train, validation, test sets, and a system prompt.

    Returns:
        train (pandas.DataFrame): The training set.
        validation (pandas.DataFrame): The validation set.
        test (pandas.DataFrame): The test set.
        system_prompt (str): The system prompt for the math problems.
    """
    dataset = load_dataset('openai/gsm8k', 'main')
    train = parse_gsm8k_subset(pd.DataFrame(dataset['train']))
    test = parse_gsm8k_subset(pd.DataFrame(dataset['test']))
    train = train.sample(frac=1, random_state=0).reset_index(drop=True)
    validation = train.iloc[1500:3000]
    train = train.iloc[:1500]    
    system_prompt = "The following are math problems. Please provide the answer to the math problem by thinking step-by-step. Finish your answer with the answer in the format \"#### X\" where X is the correct integer answer."
    return train, validation, test, system_prompt

def main(models, dataset, output_folder, num_fewshot=3, api='together', max_samples=None, sequential=False):
    """
    Run the main function for API querying
    Args:
        models (list): List of models to run.
        dataset (str): Dataset to use for model selection.
        output_folder (str): Output folder to store the results.
        num_fewshot (int, optional): Number of few-shot examples to use. Defaults to 3.
        api (str, optional): API to use for model selection. Defaults to 'together'.
        max_samples (int, optional): Maximum number of samples to use from the dataset. Defaults to None.
    Raises:
        ValueError: If the dataset is not supported.
    Returns:
        None
    """
    if dataset == 'mmlu':
        train, validation, test, system_prompt = parse_mmlu()
        parser = mmlu_parser
    elif dataset == 'gsm8k':
        train, validation, test, system_prompt = parse_gsm8k()
        parser = gsm8k_parser
    else:
        raise ValueError(f'Dataset {dataset} not supported')
    if max_samples is not None:
        train = train.iloc[:max_samples]
        test = test.iloc[:max_samples]
    
    for df_name, df in zip(['train', 'test'], [train, test]):
        queries = get_queries(df, validation, num_fewshot, system_prompt)
        for model in models:
            if not os.path.isfile(f'{output_folder}/{df_name}/{model["name"]}.json'):
                api_here = api if not model.get('is_huggingface', False) else 'huggingface'
                model_answers, costs, qualities = run_dataset(
                    model=model['name'],
                    api=api_here,
                    df=df,
                    validation_df=validation,
                    num_fewshot=num_fewshot,
                    read_cost=model['read_cost'],
                    write_cost=model['write_cost'],
                    parse_answer_function=parser,
                    system_message=system_prompt,
                    sequential=sequential
                )
                store_model_outputs(model_answers, costs, qualities, f'{output_folder}/{df_name}/{model["name"]}.json')
            else:
                read_data = pd.read_json(f'{output_folder}/{df_name}/{model["name"]}.json')
                read_data['model_answers'] = read_data['model_answers'].apply(lambda x: [x[0], x[1], parser(x[0])])
                read_data.to_json(f'{output_folder}/{df_name}/{model["name"]}.json')
        store_all_models([f'{output_folder}/{df_name}/{model["name"]}.json' for model in models], [model['name'] for model in models], 
                         f"{output_folder}/{df_name}", queries)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mmlu')
    parser.add_argument('--output_folder', type=str, default='data/free_form')
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
    ]
    output_folder = f"{args.output_folder}/{args.dataset}"
    if args.sequential:
        output_folder += '_sequential'
    main(models, args.dataset, output_folder, 
         num_fewshot=args.num_fewshot, api='together', max_samples=args.samples, 
         sequential=args.sequential)
