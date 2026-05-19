from .classification import ClassificationQualityComputer
from .base_computer import BaseComputer
from .cost_computer import BaseCostComputer
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

class OpenFormCostComputer(BaseCostComputer):
    def __init__(self, input_costs, output_costs, tokenizers=None, tokenize=True, 
                 store_all=False, constant_cost=False, is_latency_cost=False):
        """
        Initializes an instance of the OpenFormCostComputer class.
        Computes the cost of running a model on a question.

        Parameters:
        - input_costs (list): A list of input costs per token for each model.
        - output_costs (list): A list of output costs per token for each model.
        - tokenizers (list, optional): A list of tokenizers. Defaults to None.
        - tokenize (bool, optional): A flag indicating whether to tokenize. Defaults to True.
        - store_all (bool, optional): A flag indicating whether to store all computed costs. 
                                        Defaults to False.
        - constant_cost (bool, optional): A flag indicating whether to set the computed cost to a constant for each model.
                                        Defaults to False.
        """
        super().__init__()
        self.input_costs = input_costs
        self.output_costs = output_costs
        self.tokenizers = tokenizers
        self.average_output_cost = None
        self.tokenize = tokenize
        self.computed_costs = []
        self.store_all = store_all
        self.constant_cost = constant_cost
        self.is_latency_cost = is_latency_cost
        self.latency_models = []
        self.computed_latency_costs = []
        assert tokenizers is not None or not tokenize

    def fit(self, questions, model_answers, measure):
        self.average_output_cost = []
        self.constant_costs = []
        for model in range(len(model_answers[0])):
            tokenized_answers = [model_answers[i][model][0] for i in range(len(model_answers))]
            self.computed_costs.append(dict())
            self.computed_latency_costs.append(dict())
            if self.tokenize:
                tokenized_answers = self.tokenizers[model](tokenized_answers, padding=False)['input_ids']
            average_output_cost = np.mean([self.output_costs[model] * len(tokenized_answer) for tokenized_answer in tokenized_answers])
            self.average_output_cost.append(average_output_cost / (len(model_answers)))
            tokenized_questions = questions
            if len(tokenized_questions) > 0 and not isinstance(tokenized_questions[0], str):
                tokenized_questions = [question[0] for question in tokenized_questions]
            if self.tokenize:
                tokenized_questions = self.tokenizers[model](tokenized_questions, padding=False)['input_ids']
            average_input_cost = np.mean([self.input_costs[model] * len(tokenized_question) for tokenized_question in tokenized_questions])
            self.constant_costs.append(average_input_cost + average_output_cost)
            latency_model_X = [[len(question[0])] for question in questions]
            self.latency_models.append(LinearRegression())
            self.latency_models[model].fit(latency_model_X, measure[:, model])

    def add_latency_ground_truths(self, questions, model_answers, latencies):
        """
        Add latency ground truths for the given questions and model answers.

        Args:
            questions (list): List of questions.
            model_answers (list): List of model answers.
            latencies (list): List of latencies.

        Returns:
            None
        """
        for index, question in enumerate(questions):
            for model in range(len(model_answers[0])):
                if not isinstance(question, str):
                    question = question[0]
                if question not in self.computed_costs[model]:
                    self.computed_latency_costs[model][question] = latencies[index][model]

    def predict(self, questions, model_answers):
        length_models = len(model_answers[0])

        all_costs = []
        for i in range(len(questions)):
            models_run = ','.join([str(int(model_answers[i][model] is not None)) 
                                   for model in range(length_models)])
            tokenized_question = None
            tokenized_model_answers = None
            
            costs = []
            for model in range(length_models):
                if self.constant_cost:
                    cost = self.constant_costs[model]
                    costs.append(cost)
                    continue
                question = questions[i]
                if not isinstance(question, str):
                    question = question[0]
                if self.is_latency_cost and question in self.computed_latency_costs[model] and model_answers[i][model] is not None:
                    costs.append(self.computed_latency_costs[model][question])
                    continue
                if (self.training or self.store_all) and question in self.computed_costs[model] and models_run in self.computed_costs[model][question]:
                    costs.append(self.computed_costs[model][question][models_run])
                    continue
                
                if self.is_latency_cost:
                    cost = self.latency_models[model].predict([[len(question)]])[0]
                else:
                    if tokenized_question is None:
                        tokenized_question = question
                        tokenized_model_answers = [answer[0] if answer is not None else None 
                                                for answer in model_answers[i]]
                        if self.tokenize:
                            tokenized_question = [self.tokenizers[model]([question], padding=False)['input_ids'][0] for model in range(length_models)]
                            tokenized_model_answers = [
                                self.tokenizers[model]([answer[0]], padding=False)['input_ids'][0] 
                                if answer is not None else None
                                for answer, model in zip(model_answers[i], range(length_models))
                            ]
                    cost = self.input_costs[model] * len(tokenized_question[model])
                    if model_answers[i][model] is None and models_run.count('1') == 0:
                        cost += self.average_output_cost[model]
                    elif model_answers[i][model] is None:
                        cost += self.output_costs[model] * np.mean([len(answer) 
                                                                    for answer in tokenized_model_answers 
                                                                    if answer is not None])
                    else:
                        cost += self.output_costs[model] * len(tokenized_model_answers[model])
                
                costs.append(cost)
                if self.store_all or self.training:
                    if question in self.computed_costs[model]:
                        self.computed_costs[model][question][models_run] = cost
                    else:
                        self.computed_costs[model][question] = {models_run: cost}

            all_costs.append(costs)

        return np.array(all_costs)


class OpenFormQualityComputer(ClassificationQualityComputer):
    def __init__(self, model_class=LogisticRegression, 
                 require_constant_not_run=False, 
                 question_indicator=r'Question:', answer_indicator=r'Answer:',
                 remove_options=['\nA:', '\nA.'],
                 max_depth=None, n_samples=100, store_all=False, **kwargs):
        """
        Initializes the OpenFormQualityComputer class.
        For a description of all parameters, we refer to the ClassificationQualityComputer class.
        """
        super().__init__(
            model_class=model_class,
            require_constant_not_run=require_constant_not_run,
            question_indicator=question_indicator,
            answer_indicator=answer_indicator,
            remove_options=remove_options,
            max_depth=max_depth,
            n_samples=n_samples,
            store_all=store_all,
        )

    def parse_question(self, question, remove_options=True):
        if not isinstance(question, str):
            question = question[0]
        question = question.split(self.question_indicator)[-1]
        if self.remove_options is not None and remove_options:
            for option in self.remove_options:
                question = question.split(option)[0].strip()
        question = question.split(self.answer_indicator)[0].strip()
        return question
    
    def agreement_features(self, question, n_models, models_answers_sample):
        features = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if models_answers_sample[i] is not None and models_answers_sample[j] is not None:
                    features.append(models_answers_sample[i][2] == models_answers_sample[j][2])
        return features

    def certainty_features(self, model, models_answers_sample):
        if models_answers_sample[model] is None:
            return []
        else:
            logprobs = models_answers_sample[model][1]
            if len(logprobs) == 0:
                return [0 for _ in range(8)]
            return [np.sum(logprobs) / 100, np.mean(logprobs), 
                    np.min(logprobs), np.median(logprobs),
                    np.quantile(logprobs, 0.25), np.quantile(logprobs, 0.1), 
                    np.log(len(logprobs)), 
                    int(models_answers_sample[model][2] is not None)]