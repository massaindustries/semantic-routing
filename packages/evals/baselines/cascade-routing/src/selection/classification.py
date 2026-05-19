from .base_computer import BaseComputer
from .quality_computer import BaseQualityComputer
from .cost_computer import BaseCostComputer
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import json
from tqdm import tqdm
import os

class ClassificationCostComputer(BaseCostComputer):
    def __init__(self, input_costs, output_costs, tokenizers=None, 
                 tokenize=True, n_output_tokens=1, 
                 constant_cost=False, store_all=False, 
                 is_latency_cost=False):
        """
        Initialize the Classification Computer object.

        Args:
            input_costs (list): The input costs per token for each model.
            output_costs (list): The output costs per token for each model.
            tokenizers (list, optional): The tokenizers for each model. Defaults to None.
            tokenize (bool, optional): Whether to tokenize. Defaults to True.
            n_output_tokens (int, optional): The number of output tokens. Defaults to 1.
            constant_cost (bool, optional): Whether to always output constant costs for each model. Defaults to False.
            store_all (bool, optional): Whether to store all predictions. Speeds up prediction at the cost of memory. 
                                        Defaults to False.
        """
        super().__init__()
        self.input_costs = input_costs
        self.output_costs = output_costs
        self.tokenizers = tokenizers
        self.tokenize = tokenize
        self.n_output_tokens = n_output_tokens
        self.constant_cost = constant_cost
        self.store_all = store_all
        self.is_latency_cost = is_latency_cost
        self.latency_models = []
        self.computed_latency_costs = []
        self.computed_costs = []
        assert tokenizers is not None or not tokenize

    def fit(self, questions, model_answers, measure):
        self.constant_costs = []
        for model in range(len(model_answers[0])):
            self.constant_costs.append(
                np.mean(measure[:, model])
            )
            self.computed_costs.append(dict())
            self.computed_latency_costs.append(dict())
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
        for model in range(length_models):
            costs = []
            for index, question in enumerate(questions):
                models_run = ','.join([str(int(model_answers[index][model_] is not None)) 
                                   for model_ in range(length_models)])
                if not isinstance(question, str):
                    question = question[0]
                if self.is_latency_cost and question in self.computed_latency_costs[model] and model_answers[index][model] is not None:
                    costs.append(self.computed_latency_costs[model][question])
                    continue
                if (self.training or self.store_all) and question in self.computed_costs[model] and models_run in self.computed_costs[model][question]:
                    costs.append(self.computed_costs[model][question][models_run])
                    continue
                elif not self.tokenize:
                    tokenized_question = question
                else:
                    tokenized_question = self.tokenizers[model]([question], padding=False)['input_ids'][0]

                if self.constant_cost:
                    cost = self.constant_costs[model]
                else:
                    cost = self.input_costs[model] * len(tokenized_question)
                    cost += self.output_costs[model] * self.n_output_tokens # one output token
                if self.is_latency_cost:
                    cost = self.latency_models[model].predict([[len(question)]])[0]
                costs.append(cost)
                if self.training or self.store_all:
                    if question in self.computed_costs[model]:
                        self.computed_costs[model][question][models_run] = cost
                    else:
                        self.computed_costs[model][question] = {models_run: cost}

            all_costs.append(costs)
        return np.array(all_costs).T
    

class ClassificationQualityComputer(BaseQualityComputer):
    def __init__(self, model_class=LogisticRegression, 
                 n_highest_include=1, require_constant_not_run=False, 
                 question_indicator=r'Question:', answer_indicator=r'Answer:', 
                 remove_options=['\nA:', '\nA.'], 
                 add_entropy=True, add_js_divergence=True, 
                 add_equal_argmax=True, max_depth=None, 
                 n_samples=100, store_all=False):
        """
        Initializes the ClassificationSelection object.

        Args:
            model_class (class): The class of the prediction model to be used. Default is LogisticRegression.
            n_highest_include (int): The number of highest class probabilities to include in the features. 
                                    Default is 1.
            require_constant_not_run (bool): Whether to require constant predictions for uncomputed models. 
                                            Default is False.
            question_indicator (str): The indicator for the "Question" part of a classification question. 
                                        Used for filtering and computing question length.
                                        Default is 'Question:'.
            answer_indicator (str): The indicator for the "Answer" part of the classification question. 
                                    Used for filtering and computing question length.
                                    Default is 'Answer:'.
            remove_options (list): The indicators of the options to remove from the text. 
                                    Default is ['\nA:', '\nA.'].
            add_entropy (bool): Whether to add entropy as feature. Default is True.
            add_js_divergence (bool): Whether to add Jensen-Shannon divergence between model answers as feature. Default is True.
            add_equal_argmax (bool): Whether to add equal prediction between model answers as feature. Default is True.
            max_depth (int): The maximum depth for the cascade router. Default is None.
            n_samples (int): The number of samples to compute max(q_1, ..., q_n). Default is 100.
            store_all (bool): Whether to store all results. 
                            Speeds up prediction at the cost of memory.
                            Default is False.
        """
        super().__init__(n_samples=n_samples)
        self.model_class = model_class
        self.models = None
        self.n_highest_include = n_highest_include
        self.sigma_per_n_models_run = None
        self.require_constant_not_run = require_constant_not_run
        self.constant_qualities = []
        self.question_indicator = question_indicator
        self.answer_indicator = answer_indicator
        self.remove_options = remove_options
        self.add_entropy = add_entropy
        self.add_js_divergence = add_js_divergence
        self.add_equal_argmax = add_equal_argmax
        self.max_depth = max_depth
        self.store_all = store_all
        self.lookup_embeddings = None
        self.predict_proba = hasattr(self.model_class(), 'predict_proba')
        self.question_predictions = dict()

    @property
    def is_independent(self):
        return False

    def entropy(self, p):
        """
        Calculate the entropy of a probability distribution.

        Args:
            p (numpy.ndarray): The probability distribution.

        Returns:
        float: The entropy value.
        """
        return -np.sum(p * np.log2(np.maximum(p, 1e-16)))
    
    def kl_divergence(self, p, q):
        """
        Calculates the Kullback-Leibler divergence between two probability distributions.

        Args:
            p (numpy.ndarray): The first probability distribution.
            q (numpy.ndarray): The second probability distribution.

        Returns:
            float: The Kullback-Leibler divergence between p and q.
        """
        return np.sum(p * np.log2(np.maximum(p, 1e-16) / np.maximum(q, 1e-16)))
    
    def js_divergence(self, p, q):
        """
        Calculates the Jensen-Shannon divergence between two probability distributions.

        Args:
            p: numpy array or list, representing the first probability distribution.
            q: numpy array or list, representing the second probability distribution.

        Returns:
            js_div: float, the Jensen-Shannon divergence between p and q.
        """
        m = (p + q) / 2
        return (self.kl_divergence(p, m) + self.kl_divergence(q, m)) / 2
    
    def parse_question(self, question, remove_options=True):
        """
        Parses the given question and returns the extracted question text.

        Args:
            question (str or list): The question to be parsed. 
                                    If a list is provided, the first element will be used.
            remove_options (bool): Flag indicating whether to remove options from the question. 
                                    Default is True.

        Returns:
            str: The extracted question text.
        """
        if not isinstance(question, str):
            question = question[0]
        question = question.split(self.question_indicator)[-1]
        if self.remove_options is not None and remove_options:
            for remove_option in self.remove_options:
                question = question.split(remove_option)[0].strip()
        question = question.split(self.answer_indicator)[0].strip()
        return question

    def fit(self, questions, model_answers, measure):
        n_models = len(model_answers[0])
        self.models = [dict() for _ in range(n_models)]
        X, X_all_models, y, for_model, all_models_run = self.prepare_data(questions, model_answers, measure, n_models)
        y_pred_all = np.zeros((len(X) // n_models, n_models))
        y_pred_all_models = np.zeros((len(X) // n_models, n_models))
        for model in range(n_models):
            models_to_fit = np.unique(all_models_run)
            for models_run_string in tqdm(models_to_fit, desc=f'Model {model}'):
                self.models[model][models_run_string] = self.model_class()
                indices_run = [i for i in range(len(X)) 
                                if all_models_run[i] == models_run_string and for_model[i] == model]
                X_here = np.array([X[i] for i in indices_run])
                y_here = np.array([y[i] for i in indices_run])
                self.models[model][models_run_string].fit(X=X_here, y=y_here)
                if self.predict_proba:
                    y_pred = self.models[model][models_run_string].predict_proba(X_here)[:, 1]
                else:
                    y_pred = self.models[model][models_run_string].predict(X_here)
                
                indices_pred_all = [i // n_models for i in indices_run]
                y_pred_all[indices_pred_all, model] = y_pred

            indices_all = [i for i in range(len(X)) if for_model[i] == model]
            y_pred_all_models[:, model] = self.predict_model(self.models[model][','.join([str(i) for i in range(n_models)])], X_all_models[indices_all])

        self.compute_sigma(n_models,  all_models_run, y_pred_all, 
                           y_pred_all_models, models_to_fit)

    def compute_sigma(self, n_models, all_models_run, 
                      y_pred_all, y_pred_all_models, models_to_fit):
        """
        Compute the deviation of the predicted values from the actual values.

        Parameters:
            n_models (int): The number of models.
            all_n_models_run (numpy.ndarray): Array containing the number of models run for each iteration.
            all_models_run (numpy.ndarray): Array containing the models run for each iteration.
            y_pred_all (numpy.ndarray): Array containing the predicted values for all iterations.
            y_pred_all_models (numpy.ndarray): Array containing the predicted values for all models and iterations.
            models_to_fit (list): List of models to fit.
        """
        all_models_run_single = np.array([all_models_run[i] 
                                          for i in range(0, len(all_models_run), n_models)])
        self.sigma_per_n_models_run = dict()
        for models_run_string in models_to_fit:
            diff = y_pred_all[all_models_run_single == models_run_string] - y_pred_all_models[all_models_run_single == models_run_string]
            self.sigma_per_n_models_run[models_run_string] = np.cov(diff.T)

    def prepare_data(self, questions, model_answers, measure, n_models):
        """
        Prepare the data for fitting.
        Args:
            questions (list): List of questions.
            model_answers (list): List of model answers.
            measure (list): List of measures.
            n_models (int): Number of models.
        Returns:
            tuple: A tuple containing the following arrays:
                - X (ndarray): Input data for each model.
                - X_all_models (ndarray): Input data for all models.
                - y (ndarray): Output data.
                - for_model (ndarray): Model index for each data point.
                - all_models_run (ndarray): String representation of models used for each data point.
        """
        X = []
        X_all_models = []
        y = []
        for_model = []
        all_models_run = []

        for model in range(n_models):
            self.constant_qualities.append(np.mean([measure[i][model] for i in range(len(questions))]))

        for i in range(len(questions)):
            for n_models_run in range(n_models + 1):
                if self.max_depth is not None and n_models > n_models_run > self.max_depth:
                    continue
                for models_run in itertools.combinations(range(n_models), n_models_run):
                    models_run_string = ','.join([str(model) for model in sorted(models_run)])
                    
                    models_answers_sample = [answer if model in models_run else None 
                                             for model, answer in enumerate(model_answers[i])]
                    measure_sample = measure[i]
                    
                    for model in range(n_models):
                        X_sample, y_sample = self.generate_sample_input_output(questions[i], model, 
                                                                               n_models, 
                                                                               models_answers_sample, 
                                                                               measure_sample, i)
                        X_sample_all_models, _ = self.generate_sample_input_output(questions[i], model, 
                                                                                   n_models, 
                                                                                   model_answers[i], 
                                                                                   measure_sample, i)
                        y.append(y_sample)
                        X.append(X_sample)
                        X_all_models.append(X_sample_all_models)
                        all_models_run.append(models_run_string)
                        for_model.append(model)

        X_all_models = np.array(X_all_models)
        y = np.array(y)
        all_models_run = np.array(all_models_run)
        for_model = np.array(for_model)
        return X,X_all_models,y,for_model,all_models_run
    
    def predict_model(self, model, X):
        """
        Predicts the target variable using the given model.

        Parameters:
            model (object): The trained model used for prediction.
            X (array-like): The input features for prediction.

        Returns:
            array-like: The predicted target variable values.
        """
        if self.predict_proba:
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)


    def predict(self, questions, model_answers):
        n_models = len(model_answers[0])
        n_models_answered = np.array([
            len([model_answer for model_answer in model_answers[i] if model_answer is not None]) 
            for i in range(len(questions))
        ])
        all_models_run_strings = np.array([','.join([str(i) for i in range(n_models) 
                                                     if model_answers[j][i] is not None]) 
                                            for j in range(len(questions))]) 
        y = np.zeros((len(questions), n_models))

        for model in range(n_models):
            y_model_done = np.zeros(len(questions)).astype(bool)
            if self.training or self.store_all:
                for i in range(len(questions)):
                    models_run = all_models_run_strings[i]
                    question = questions[i]
                    if not isinstance(question, str):
                        question = question[0]
                    question_prediction = self.question_predictions.get(model, 
                                                                        dict()).get(models_run, dict()).get(question, None)
                    if question_prediction is not None:
                        y[i, model] = question_prediction
                        y_model_done[i] = True
            
            y_model = np.zeros(np.count_nonzero(np.logical_not(y_model_done)))
            X_model = [self.generate_sample_input_output(questions[i], model, 
                                                         n_models, model_answers[i])[0] 
                        for i in range(len(questions)) if not y_model_done[i]]
            model_answers_here = [model_answers[i] for i in range(len(questions)) if not y_model_done[i]]
            models_run_strings = all_models_run_strings[np.logical_not(y_model_done)]
            for models_run_string in self.models[model].keys():
                indices = np.where(models_run_string == models_run_strings)[0]
                X = [X_model[i] for i in indices]
                if len(indices) == 0:
                    continue
                y_model[indices] = self.predict_model(self.models[model][models_run_string], X)

            if self.require_constant_not_run:
                for i in range(len(y_model)):
                    if model_answers_here[i][model] is None:
                        y_model[i] = self.constant_qualities[model]
            
            y[np.logical_not(y_model_done), model] = y_model

            if self.training or self.store_all:
                for i in range(len(questions)):
                    models_run = all_models_run_strings[i]
                    if model not in self.question_predictions:
                        self.question_predictions[model] = dict()
                    if models_run not in self.question_predictions.get(model, dict()):
                        self.question_predictions[model][models_run] = dict()
                    
                    question = questions[i]
                    if not isinstance(question, str):
                        question = question[0]
                    self.question_predictions[model][models_run][question] = y[i, model]
        return y, np.array([self.sigma_per_n_models_run[all_models_run_strings[i]] 
                                for i in range(len(questions))])

    def predict_n_answers(self, model_answers, n_models_answered, model, y_model, X_model, n_answers, model_answered):
        """
        Predicts the answers for a given model and number of answers.
        Args:
            model_answers (list): List containing model answers.
            n_models_answered (numpy.ndarray): Array of the number of models answered for each question.
            model (int): Index of the model to predict the answers for.
            y_model (numpy.ndarray): Array of the model answers.
            X_model (list): List of input features for the model.
            n_answers (int): Number of answers to predict.
            model_answered (bool): Flag indicating whether the model has already answered.
        """
        
        if model_answered:
            indices = np.where(np.logical_and(n_models_answered == n_answers, 
                                              [answer[model] is not None for answer in model_answers]))[0]
        else:
            indices = np.where(np.logical_and(n_models_answered == n_answers, 
                                              [answer[model] is None for answer in model_answers]))[0]
        X = [X_model[i] for i in indices]
        if len(indices) > 0:
            y_model[indices] = self.predict_model(self.models[model][n_answers][model_answered], X)

    def base_features(self, question, index, model):
        """
        Generate a list of base features for a given question, index, and model.

        Parameters:
           question (str or tuple): The question to generate features for. 
                                    If a tuple is provided, the first element is the question string 
                                    and the remaining elements are additional features.
            index (int or None): The index of the question in the training dataset. 
                                If None, the question is not in the dataset.
            model (str): The name of the model.

        Returns:
            features (list): A list of features for the given question, index, and model.
        """
        features = []
        if not isinstance(question, str):
            question, additional_features = question[0], question[1:]
            features.extend(additional_features)

        question_here = self.parse_question(question, remove_options=False)
        n_options = sum([f'\n{x}:' in question or f'\n{x}.' in question_here for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
        features.append(1 / (max(n_options, 1)))
        return features

    def agreement_features(self, question, n_models, models_answers_sample):
        """
        Calculates agreement features between models' answers.

        Args:
            n_models (int): The number of models.
            models_answers_sample (list): A list of models' answers.

        Returns:
            list: A list of agreement features.

        """
        features = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if models_answers_sample[i] is not None and models_answers_sample[j] is not None:
                    if self.add_js_divergence:
                        features.append(self.js_divergence(models_answers_sample[i], 
                                                           models_answers_sample[j]))
                    if self.add_equal_argmax:
                        features.append(float(np.argmax(models_answers_sample[i]) == np.argmax(models_answers_sample[j])))
        return features

    def certainty_features(self, model, models_answers_sample):
        """
        Calculate the certainty features for a given model and models_answers_sample.

        Parameters:
        - model: The index of the model for which to calculate the certainty features.
        - models_answers_sample: A list of model answers for each model.

        Returns:
        - A list of certainty features for the given model.

        Raises:
        - None.

        """
        if models_answers_sample[model] is None:
            return []
        else:
            model_answer_highest = sorted(models_answers_sample[model], key=lambda x: -x)[:self.n_highest_include]
            if len(model_answer_highest) < self.n_highest_include:
                for _ in range(self.n_highest_include - len(model_answer_highest)):
                    model_answer_highest.append(0)
            if self.add_entropy:
                model_answer_highest.append(self.entropy(models_answers_sample[model]))
            return model_answer_highest

    def generate_sample_input_output(self, question, model, n_models, models_answers_sample, 
                                     measure_sample=None, index=None):
        """
        Generates a sample input and output for model selection.

        Args:
            question (str): The question for which the sample input and output are generated.
            model (int): The index of the model being evaluated.
            n_models (int): The total number of models.
            models_answers_sample (list): A list of model answers for the sample.
            measure_sample (list, optional): A list of measures for the sample. Defaults to None.
            index (int, optional): The index of the question. Defaults to None.

        Returns:
            tuple: A tuple containing the sample input and output.
        """
        X_sample = []
        X_sample += self.base_features(question, index, model)
        X_sample += self.agreement_features(question, n_models, models_answers_sample)
        X_sample += self.certainty_features(model, models_answers_sample)
        if len(X_sample) == 0:
            X_sample = [0]

        if measure_sample is not None:
            return X_sample, measure_sample[model]
        return X_sample, None
