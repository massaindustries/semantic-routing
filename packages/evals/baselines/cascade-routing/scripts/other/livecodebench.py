import pandas as pd
import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm

base_folder = "output"

model_names = [
    "Qwen2.5-Coder-Ins-1.5B",
    "Qwen2.5-Coder-Ins-7B",
    "Qwen2.5-Math-Ins-1.5B",
    "Qwen2.5-Math-Ins-7B",
]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

results = dict()
qualities = pd.DataFrame()
model_answers = pd.DataFrame()
costs = pd.DataFrame()
queries = []

for model_name in model_names:
    qualities_model = []
    model_answers_model = []
    costs_model = []
    queries_model = []
    data = json.load(open(os.path.join(base_folder, model_name, "Scenario.codegeneration_10_0.2_eval_all.json")))
    
    for query in tqdm(data):
        queries_model.append([
            query["question_content"],
            query["platform"],
            query["difficulty"]
        ])
        output = query["output_list"][0]
        quality = query["graded_list"][0]
        answer = json.loads(query["metadata"][0])
        if "output" in answer:
            answer = answer["output"]
        else:
            answer = "correct" # just for matching purposes, will not be used in the evaluation
        tokenized_length = len(tokenizer.encode(output)) + len(tokenizer.encode(query["question_content"]))
        cost = tokenized_length * 7 if "7B" in model_name else tokenized_length * 1.5
        qualities_model.append(quality)
        model_answers_model.append([output, cost, answer])
        costs_model.append(cost)
    
    qualities[model_name] = qualities_model
    model_answers[model_name] = model_answers_model
    costs[model_name] = costs_model
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
