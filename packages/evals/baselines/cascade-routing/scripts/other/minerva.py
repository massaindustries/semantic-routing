import pandas as pd
import numpy as np
import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import re
base_folder = "data"

model_names = [
    "Qwen__Qwen2.5-Coder-1.5B-Instruct",
    "Qwen__Qwen2.5-Coder-7B-Instruct",
    "Qwen__Qwen2.5-Math-1.5B-Instruct",
    "Qwen__Qwen2.5-Math-7B-Instruct",
]

def change_name(name):
    return name.split("__")[1].replace("Instruct", "Ins")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

results = dict()
qualities = pd.DataFrame()
model_answers = pd.DataFrame()
costs = pd.DataFrame()
queries = []

def read_file(name):
    data = []
    with open(name, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        data.append(result)
    return data

def extract_last_boxed(output):
    # find \boxed{ANSWER} since ANSWER can contain {} we need to count the number of { and } to find the end
    # of the answer
    start = output.rfind("\\boxed{")
    if start == -1:
        if "final answer is" in output:
            find_numb = output.split("final answer is")[-1].split(".")
            # remove all non-numeric characters
            find_numb = re.sub(r'[^\d.]', '', find_numb[0])
            if find_numb != "":
                return find_numb
            return None
        return None
    end = start + len("\\boxed{")
    if end >= len(output):
        return None
    count = 1
    while count > 0:
        if output[end] == "{":
            count += 1
        elif output[end] == "}":
            count -= 1
        end += 1
        if end >= len(output):
            break
    return output[start + len("\\boxed{"):end - 1]

for model_name in model_names:
    qualities_model = []
    model_answers_model = []
    costs_model = []
    queries_model = []
    filenames = os.listdir(os.path.join(base_folder, model_name))
    # keep the filename with "math_algebra"
    filename = [f for f in filenames if "math_algebra" in f][0]
    data = read_file(os.path.join(base_folder, model_name, filename))
    
    for query in tqdm(data):
        queries_model.append([
            query["doc"]["problem"],
            query["doc"]["type"],
            query["doc"]["level"]
        ])
        output = query["resps"][0][0]
        quality = query["exact_match"]
        # search for "\boxed{ANSWER}"
        extracted_answer = extract_last_boxed(output)
        if extracted_answer is not None and "=" in extracted_answer:
            extracted_answer = extracted_answer.split("=")[-1]
        tokenized_length = len(tokenizer.encode(output)) + len(tokenizer.encode(query["doc"]["problem"]))
        cost = tokenized_length * 7 if "7B" in model_name else tokenized_length * 1.5
        qualities_model.append(quality)
        model_answers_model.append([output, cost, extracted_answer])
        costs_model.append(cost)
    
    name_changed = change_name(model_name)
    qualities[name_changed] = qualities_model
    model_answers[name_changed] = model_answers_model
    costs[name_changed] = costs_model
    queries = queries_model

print(qualities.mean())
print(costs.mean())

os.makedirs("results", exist_ok=True)
qualities.to_json("results/qualities.json")
model_answers.to_json("results/model_answers.json")
costs.to_json("results/costs.json")
pd.DataFrame(queries).to_json("results/queries.json")
        
qualities["any"] = qualities.any(axis=1)
print(qualities["any"].mean())
