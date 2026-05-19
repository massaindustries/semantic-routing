from .cost_computer import BaseCostComputer
from .quality_computer import BaseQualityComputer
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from itertools import combinations

class SWEBenchCostComputer(BaseCostComputer):
    def __init__(self, store_all=False, constant_cost=False):
        """
        Initializes an instance of the SWEBenchCostComputer class.
        Computes the cost of running a model on a question.

        Parameters:
        - store_all (bool, optional): A flag indicating whether to store all computed costs.
        - constant_cost (bool, optional): A flag indicating whether to set the computed cost to a constant for each model.
                                        Defaults to False.
        """
        super().__init__()
        self.store_all = store_all
        self.constant_cost = constant_cost
        self.prediction_models = []

    def fit(self, questions, model_answers, measure):
        for i, model in enumerate(model_answers[0]):
            prediction_models = dict()
            for n_models_run in range(1, len(model_answers[0]) + 1):
                for combination in combinations(range(len(model_answers[0])), n_models_run):
                    if i in combination:
                        continue
                    measure_x = np.array([[float(model_answers[j][k][-1]) for k in combination] for j in range(len(questions))])
                    measure_y = measure[:, i]
                    prediction_model = LinearRegression()
                    prediction_model.fit(measure_x, measure_y)
                    prediction_models[combination] = prediction_model
            lengths = [len(question[1]) / 1000 for question in questions]
            y = measure[:, i]
            X = np.array(lengths).reshape(-1, 1)
            prediction_model_length = LinearRegression()
            prediction_model_length.fit(X, y)
            self.prediction_models.append((prediction_model_length, prediction_models))

    def predict(self, questions, model_answers):
        length_models = len(model_answers[0])

        all_costs = []
        for i in range(len(questions)):
            costs = []
            models_run = [j for j in range(length_models) if model_answers[i][j] is not None]
            # sort models_run
            models_run = sorted(models_run)
            for j in range(length_models):
                if j in models_run:
                    costs.append(float(model_answers[i][j][-1]))
                elif any([model_answers[i][other_model] is not None for other_model in range(length_models)]):
                    features = [float(model_answers[i][other_model][-1]) for other_model in models_run]
                    costs.append(float(self.prediction_models[j][1][tuple(models_run)].predict([features])[0]))
                else:
                    length_q = len(questions[i][1]) / 1000
                    costs.append(float(self.prediction_models[j][0].predict([[length_q]])[0]))
            all_costs.append(costs)
        return np.array(all_costs)

class SWEBenchQualityComputer(BaseQualityComputer):
    def __init__(self, max_depth=None, n_samples=100,
                 access_tests=True,
                 require_constant_not_run=False, 
                 repo_names=["django/django", 
                             "sympy/sympy",
                             "astropy/astropy",
                             "psf/requests", 
                             "pytest-dev/pytest",
                             "pylint-dev/pylint",
                             "sphinx-doc/sphinx",
                             "scikit-learn/scikit-learn",
                             "matplotlib/matplotlib",
                             "pydata/xarray",
                             "pallets/flask",
                             "mwaskom/seaborn"]):
        """
        Initialize the SweBench Quality Computer class.

        Parameters:
            max_depth (int, optional): The maximum depth of the model. Defaults to None.
            n_samples (int, optional): The number of samples to use for computed expected value of max. Defaults to 100.
            require_constant_not_run (bool, optional): Flag to require constant not run. Defaults to False.
            repo_names (list of str, optional): List of repository names in the benchmark. Defaults to a predefined list of popular repositories.

        Attributes:
            max_depth (int): The maximum depth of the model.
            repo_names (list of str): List of repository names to use.
            require_constant_not_run (bool): Flag to require constant not run.
            prediction_models (list): List to store prediction models.
            variances (list): List to store variances.
        """
        super().__init__(
            n_samples=n_samples,
        )
        self.max_depth = max_depth
        self.repo_names = repo_names
        self.require_constant_not_run = require_constant_not_run
        self.prediction_models = []
        self.variances = []
        self.access_tests = access_tests

    def fit(self, questions, model_answers, measure):
        for i, model in enumerate(model_answers[0]):
            X = [self.base_features(question) for question in questions]
            y = measure[:, i]
            linear = LogisticRegression(max_iter=5000)
            linear.fit(X, y)
            self.prediction_models.append(linear)
            self.variances.append(np.var(y - linear.predict_proba(X)[:, 1]))

    def predict(self, questions, model_answers):
        length_models = len(model_answers[0])

        all_qualities = []
        all_variances = []
        for i in range(len(questions)):
            qualities = []
            variances = np.zeros((length_models, length_models))
            for j in range(length_models):
                if model_answers[i][j] is not None and self.access_tests:
                    result = float(model_answers[i][j][0])
                    qualities.append(result)
                    variances[j][j] = 1e-6
                else:
                    features = self.base_features(questions[i])
                    qualities.append(self.prediction_models[j].predict_proba([features])[0, 1])
                    variances[j][j] = self.variances[j]
            all_qualities.append(qualities)
            all_variances.append(variances)

        return np.array(all_qualities), np.array(all_variances)

    def base_features(self, question):
        """
        Generate a list of base features for a given question, index, and model.

        Parameters:
           question (str or tuple): The question to generate features for. 
                                    If a tuple is provided, the first element is the question string 
                                    and the remaining elements are additional features.
        Returns:
            features (list): A list of features for the given question, index, and model.
        """
        if self.require_constant_not_run:
            return [1]
        features = [len(question[1]) / 1000]
        for repo_name in self.repo_names:
            features.append(int(repo_name.replace("/", "__") in question[0]))
        return features