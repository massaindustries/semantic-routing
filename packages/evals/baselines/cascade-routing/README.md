# A Unified Approach to Routing and Cascading for LLMs

This repository provides the code for our paper *"A Unified Approach to Routing and Cascading for LLMs"*. In this README, we guide you through installing the library, using it effectively, and reproducing our results.

## Installation

Follow these steps to set up the environment and install the necessary dependencies. Make sure you have [Conda](https://anaconda.org/anaconda/conda) installed.

1. **Clone the repository:**
```bash
git clone https://github.com/eth-sri/cascade-routing.git
cd cascade-routing
```
2. **Create and activate the enviroment**
```bash
conda create -n selection python=3.11
conda activate selection
```
3. **Install the package in editable mode:**
```bash
python -m pip install -e .
```
4. **(Optional) Install specific package versions to match our setup:**
```bash
python -m pip install -r requirements.txt
```

## Getting Started

This library allows you to run cascading, routing, or cascade routing strategies for any downstream application. However, applying these model selection strategies to your specific use case requires some additional custom code. There are a few key steps involved in this process:

1. **Define Representations for Your Input Queries and Model Responses**
2. **Write Your Quality and Cost Estimators**
3. **Run Queries Through All Models**
4. **Fit the Cascade Router**
5. **Apply Cascade Routing to New Queries**

Since steps (1) and (2) are highly dependent on your specific application, this repository provides only example implementations. Below, we explain each step in more detail and outline how to create your own quality and cost estimators.

### Step 1. Define Representations for Your Input Queries and Model Responses
Start by determining the representation format for both the input queries and model responses. For example, a basic setup might treat both queries and responses as strings. Alternatively, a query might also include log probabilities, with each representation structured like this: `('Hello world', [-6.333, -2.1111])`.

These representations are essential because they will be used as inputs for your quality and cost estimations. This library is flexible and allows for any kind of representation you need.

### Step 2. Write Your Quality and Cost Estimators
Next, create your custom quality and cost estimators. Both should be implemented as classes that include a `predict` function. Both the quality and cost estimators should extend from the provided base classes:

```python
from selection import BaseCostComputer, BaseQualityComputer

class YourCostComputer(BaseCostComputer):
    def predict(self, questions, model_answers):
        ...

class YourQualityComputer(BaseQualityComputer):
    def predict(self, questions, model_answers):
        ...
```

The `predict` function takes two inputs:
- `questions`: A list of queries using your chosen representation format.
- `model_answers`: A list of lists, where `model_answers[i][j]` contains the response from model $j$ to question $i$. If the model hasn't been selected to answer a specific question yet, the corresponding value will be None. Thus, your quality and cost computers should be able to handle these None values and still return an estimate.

For example:
```python
questions = [
    ('Hello world', [-6.333, -2.1111]), 
    ('Cascade Routing', [-3.222, -1.1111])
]
model_answers = [
    [None, None], # no model selected for the first question yet
    ['Great idea', None] # first model responded to the second question
]
```

**Cost Estimator**
Your cost estimator should return a list of costs for each model applied to each question, like so:
```python
output = cost_estimator.predict(questions, model_answers)
print(output)
# Output: 
[
    [1.43, 2.6] # Cost for question 1 for each model
    [1.1, 2.4] # Cost for question 2 for each model
]
```
**Quality Estimator:**  For the quality estimation, return the quality score and the uncertainty for each estimate (as a covariance matrix). If you're not handling uncertainty, you can return None in place of the covariance matrix. Here's an example:
```python
output = cost_estimator.predict(questions, model_answers)
print(output)
# Output:
[
    [0.5, 0.9] # Quality for question 1
    [0.4, 0.8] # Quality for question 2
], [
    np.array([0.1, 0.01], [0.01, 0.1]), # Covariance for question 1
    None # No covariance for question 2
]
```
We have implemented a default strategy to return these covariances that works well for most use cases. To enable this default strategy, you should fit the covariances on some training data (a couple hundred samples should suffice):
 ```python
# assumes that model_answers contains no None values
quality_estimator.fit_covariances(questions, model_answers)
```

In the `predict` method, you can then return the covariance estimates:
```python
def predict(self, questions, model_answers):
    ...
    return your_predictions, self.predict_covariances(questions, model_answers)
```



Examples of simple estimators can be found in `src/selection/quality_computer.py` and `src/selection/cost_computer.py`, while more complex ones (i.e., for classification and free-form tasks) are located in `src/selection/classification.py` and `src/selection/open_form.py`.

### Step 3. Run Queries Through All Models
Before applying cascade routing, you’ll need to run queries from your training data through all models to optimize hyperparameters and fit your quality and cost estimators. Ideally, you should split your data between fitting the estimators and optimizing hyperparameters, to avoid overfitting the hyperparameters on the estimators for their training data.

You can either run the queries using your own inference framework or use the `APIQuery` class to handle this for you. Here’s a simple example:

```python
from selection import APIQuery
import asyncio

querier = APIQuery(
    model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', # model name
    api='together', # can be ['together', 'openai', 'google', 'anthropic']
    timeout=30, # timeout on request
    temperature=0, # temperature
    max_tokens=256, # max number of generated tokens
    return_logprobs=False, # whether to return logprobs
    chat=True, # Whether the requests are formatted in chat or as completion
    max_retries=20, # How many times to retry a query before returning an error
    requests_per_second=30,  # number of requests per second
    read_cost=0.01, # cost associated with input tokens per million  
    write_cost=0.02, # cost associated with output tokens per million
)

queries = [
    [
        ('system', 'You are an assistant.'), # system message
        ('human', 'Hello'),  # human message
        ('ai', 'What can I do for you?'), # (optional) ai message
        ('human', 'Is 3.8 bigger than 3.11?') # human message
    ] # in case of completion, each query is just a string instead
]

outputs, detailed_cost, cost = asyncio.run(querier.run_queries(queries))
```
Ensure you set up the relevant API keys (i.e., ANTHROPIC_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY) for this to work. Note that this code has not been properly tested for the Google and the Anthropic API.

### Step 4. Fit the Cascade Router
Now that your queries and model responses are ready, it’s time to fit the cascade router. Below is an example using a couple of strategies to optimize hyperparameters:
```python
from selection import CascadeRouter, ConstantStrategy, HyperoptStrategy

router = CascadeRouter(
    quality_computer, cost_computer, # your quality and cost estimators
    models=['Llama-3.1-8B', 'Llama-3.1-70B'], # List of Models
    max_expected_cost=0.003, # Maximum expected cost per query
    strategies=[ # Strategies for hyperparameter optimization
        ConstantStrategy(max_lambda=10000), # Default hyperparameter strategy
        HyperoptStrategy(max_lambda=10000, n_searches=300) # Hyperopt search strategy
    ], 
    force_order=True, # Ensure models are queried in ascending order, which usually increases stability. However, you must ensure that 'models' is sorted from low to high quality.
    max_depth=None, # Maximum number of models to run on a query, setting this to 1 will route your query
    cascade=False, # setting this to True runs a cascade instead
)

queries = ... # Queries formatted as needed
model_answers = ... # Model answers formatted as needed

router.fit(
    queries, model_answers
)
```

### Step 5. Apply Cascade Routing to New Queries

Once the router is trained, you can apply it to new queries using the `predict` function. Here’s a sample workflow:

```python
models = ['Llama-3.1-8B', 'Llama-3.1-70B']
query = ('Hello world', [-6.333, -2.1111])
model_answers = [None, None] # no model responded yet

selected_model_names = router.predict([query], [model_answers])
answer = query_function(selected_model_names[0]) # Implement query_function
model_answers[models.index(selected_model_names[0])] = answer

selected_model_names = router.predict([query], [model_answers])

if selected_model_names[0] is not None: # None means to stop querying
    answer = query_function(selected_model_names[0])
    model_answers[models.index(selected_model_names[0])] = answer

selected_models = router.select_answer([query], [model_answers])
print(selected_models[0]) # Example output: 'Llama-3.1-70B'
```

## Reproducing Results

To reproduce our results, you can either start from scratch and regenerate the raw data, or download (http://files.sri.inf.ethz.ch/cascade-routing/data.tar.gz)[http://files.sri.inf.ethz.ch/cascade-routing/data.tar.gz] and extract it in this repository. Below you can find the instructions to start from scratch.

### Reproducing Results from Scratch

If you prefer to run the inference yourself, follow these steps:

1. **Obtain a Together API Key:**

   You will need a Together API key to run the inference. Once you have the key, set it as an environment variable:
   ```bash
   export TOGETHER_API_KEY="YOUR_KEY"
   ```
2. **Run the Generation Script:**
    Use the provided script to run the inference across several models and process the data:
    ```bash
    bash scripts/generation.sh
    ```
    This step will make API calls, and depending on the scale of the task, it will cost approximately 100 USD using the Together API.

To obtain the results for the SWE-Bench benchmark, first clone the Github repository [SWE-Bench/Experiments](https://github.com/swe-bench/experiments). Follow their instructions to setup AWS and allow data to be downloaded. Then, copy the files `scripts/other/swebench.sh` and `scripts/other/swebench.py` in their repo. Run:
 ```bash
python -m pip install boto3
bash swebench.sh
python swebench.py
```
These scripts will create a `data` folder in the repository. You should then copy this data folder in the `data/swebench` folder of this repository. 

To obtain the results for the LiveCodeBench benchmark, first clone their [Github repository](https://github.com/LiveCodeBench/LiveCodeBench) and follow their instructions for installation. Then, copy the files `scripts/other/livecodebench.sh` and `scripts/other/livecodebench.py` in their repo. Run:
 ```bash
bash livecodebench.sh
python livecodebench.py
``` 
These scripts will create a `data` folder in the repository. You should then copy this data folder in the `data/code_math/livecodebench` folder of this repository. 


To obtain the results for the Minerva Math benchmark, first clone the [LM Eval Harness Github repository](https://github.com/EleutherAI/lm-evaluation-harness) and follow the instructions for installation. Then, copy the files `scripts/other/minerva.sh` and `scripts/other/minerva.py` in their repo. Run:
 ```bash
bash minerva.sh
python minerva.py
```
These scripts will create a `data` folder in the repository. You should then copy this data folder in the `data/code_math/minerva` folder of this repository. 


### Running Cascade Routing and Baselines
For both options above (whether you're using pre-generated results or generating them from scratch), run the following script to perform cascade routing and evaluate its baselines on all benchmarks:
```bash
bash scripts/main.sh
```
This script runs all the experiments described in our paper. Please ensure you have sufficient CPU resources available (at least 50 cores) as the experiments are computationally intensive and may take several days to complete.

After the main script finishes, use the Jupyter notebook ([`notebooks/postprocess.ipynb`](notebooks/postprocess.ipynb)) to process the raw results and obtain the final data presented in our paper.

## Code Structure


Below is a high-level overview of the code in this repository. For detailed documentation, open the generated HTML documentation by navigating to [`docs/selection/index.html`](docs/selection/index.html) in your browser.


### Main Library ([`src/selection`](src/selection))

This directory contains the core codebase, including functions for querying models via APIs, running cascade routing, and executing the experiments described in our paper. Here are some key files of interest:

- [`src/selection/api.py`](src/selection/api.py): This file handles asynchronous API calls to models using the Langchain framework. We've adapted the Langchain code slightly in [`src/selection/completion_llms.py`](src/selection/completion_llms.py) to standardize the output format across different APIs. The API calls return not only the model output but also log probabilities (if supported by the API) and the associated query cost.

- [`src/selection/baseline_cascader.py`](src/selection/baseline_cascader.py): Implements the baseline cascading strategy, referred to as the "thresholding strategy" in our paper.

- [`src/selection/cascade_router.py`](src/selection/cascade_router.py): This file contains the main logic for our cascade router, which forms the core of our approach.

- [`src/selection/lambda_strategy.py`](src/selection/lambda_strategy.py): Includes the hyperparameter optimization strategies used in our experiments.

- [`src/selection/router.py`](src/selection/router.py): Contains the code for the baseline router. While routing can also be simulated using the cascade router code, this file includes optimizations and simplifications specific to the routing strategy.

- [`src/selection/utils.py`](src/selection/utils.py): Includes setup code for all experiments. This file is mainly relevant for reproducing results and might not be directly useful for using the library in other contexts.

### Scripts ([`scripts/`](scripts/))

The `scripts/` directory contains experiment-specific code. Each script is designed to set up the appropriate prompts and parameters for running tools from [`src/selection/utils.py`](src/selection/utils.py) in the context of specific datasets. Since these scripts are highly tailored to each experiment, we won’t give details here.

### Documentation ([`docs/selection`](docs/selection))

This folder holds the automatically generated documentation for all classes in the codebase. The documentation is generated using the `pdoc3` package based on the docstrings in the code.

### Data ([`data/`](data/))

The `data/` folder should contain all necessary data for running the experiments. Once you’ve downloaded or reproduced the data, the most relevant subfolder is [`data/results`](data/results), which contains the raw results for each experiment.

## Citation

```
@article{dekoninck2024cascaderouting, 
        title={A Unified Approach to Routing and Cascading for LLMs}, 
        author={Jasper Dekoninck and Maximilian Baader and Martin Vechev}, 
        year={2024}, 
        archivePrefix={arXiv}, 
        primaryClass={cs.CL} 
}
```