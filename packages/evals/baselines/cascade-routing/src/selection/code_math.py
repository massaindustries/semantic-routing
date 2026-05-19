from .cost_computer import BaseCostComputer
from .classification import ClassificationQualityComputer
from .quality_computer import BaseQualityComputer
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from itertools import combinations

class CodeMathCostComputer(BaseCostComputer):
    def __init__(self, store_all=False, constant_cost=False):
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
        self.store_all = store_all
        self.constant_cost = constant_cost
        self.prediction_models = []

    def fit(self, questions, model_answers, measure):
        """
        Fit the model using the provided questions, model answers, and measure.

        Parameters:
            questions (list): A list of questions where each question is a list of features.
            model_answers (list): A list of model answers where each answer is a list of tuples.
                                Each tuple contains a model identifier and a corresponding value.
            measure (numpy.ndarray): A 2D array where each row corresponds to a question and each column corresponds to a model's measure.

        Returns:
        None
        """
        for i, model in enumerate(model_answers[0]):
            prediction_models = dict()
            for n_models_run in range(1, len(model_answers[0]) + 1):
                for combination in combinations(range(len(model_answers[0])), n_models_run):
                    if i in combination:
                        continue
                    measure_x = np.array([[float(model_answers[j][k][1]) for k in combination] for j in range(len(questions))])
                    measure_y = measure[:, i]
                    prediction_model = LinearRegression()
                    prediction_model.fit(measure_x, measure_y)
                    prediction_models[combination] = prediction_model
            lengths = [len(question[0]) / 1000 for question in questions]
            y = measure[:, i]
            X = np.array(lengths).reshape(-1, 1)
            prediction_model_length = LinearRegression()
            prediction_model_length.fit(X, y)
            self.prediction_models.append((prediction_model_length, prediction_models))

    def predict(self, questions, model_answers):
        """
        Predict the costs for a given set of questions and model answers.

        Args:
            questions (list): A list of questions where each question is represented as a list of features.
            model_answers (list): A list of model answers where each answer is a list of tuples. Each tuple contains 
                                  the model's prediction and its associated cost.

        Returns:
            numpy.ndarray: A 2D array where each row corresponds to the predicted costs for each question across all models.
        """
        length_models = len(model_answers[0])

        all_costs = []
        for i in range(len(questions)):
            costs = []
            models_run = [j for j in range(length_models) if model_answers[i][j] is not None]
            # sort models_run
            models_run = sorted(models_run)
            for j in range(length_models):
                if j in models_run:
                    costs.append(float(model_answers[i][j][1]))
                elif any([model_answers[i][other_model] is not None for other_model in range(length_models)]):
                    features = [float(model_answers[i][other_model][1]) for other_model in models_run]
                    costs.append(float(self.prediction_models[j][1][tuple(models_run)].predict([features])[0]))
                else:
                    length_q = len(questions[i][0]) / 1000
                    costs.append(float(self.prediction_models[j][0].predict([[length_q]])[0]))
            all_costs.append(costs)
        return np.array(all_costs)
    

class CodeMathQualityComputer(ClassificationQualityComputer):
    def __init__(self, model_class=LogisticRegression, 
                
                 require_constant_not_run=False,
                 max_depth=None, n_samples=100, store_all=False, **kwargs):
        """
            Initialize the quality model.
            Parameters:
                model_class (class, optional): The class of the model to be used. Defaults to LogisticRegression.
                require_constant_not_run (bool, optional): Flag to require constant not run. Defaults to False.
                max_depth (int, optional): The maximum depth of the model. Defaults to None.
                n_samples (int, optional): The number of samples to be used. Defaults to 100.
                store_all (bool, optional): Flag to store all intermediate results. Defaults to False.
                **kwargs: Additional keyword arguments to be passed to the model.
        """
        super().__init__(
            model_class=model_class,
            require_constant_not_run=require_constant_not_run,
            max_depth=max_depth,
            n_samples=n_samples,
            store_all=store_all,
        )

    def parse_question(self, question, remove_options=True):
        """
        Parses the provided question and optionally removes multiple-choice options.

        This function checks if the input is a string. If not, it assumes the first element 
        of the passed list-like input is the question. Currently, the remove_options parameter 
        does not alter the returned string.

        Parameters:
            question (str or list):
                The question or a container whose first element is the question string.
            remove_options (bool):
                Whether to remove multiple-choice options from the question. Defaults to True.

        Returns:
            str: The parsed question string.
        """
        if not isinstance(question, str):
            question = question[0]
        return question
    
    def agreement_features(self, question, n_models, models_answers_sample):
        """
        Generates agreement features by comparing model answers.

        Args:
            question: The question being evaluated.
            n_models: The number of models to compare.
            models_answers_sample: A list containing the answers from each model.

        Returns:
            A list of boolean values indicating agreement between pairs of model answers.
        """
        features = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if models_answers_sample[i] is not None and models_answers_sample[j] is not None:
                    features.append(models_answers_sample[i][2] == models_answers_sample[j][2])
        return features

    def certainty_features(self, model, models_answers_sample):
        """Generate a list of certainty-based features for the given model and sample of model answers.

        Args:
            model: The model for which to generate certainty features.
            models_answers_sample: A sample of responses from the model.

        Returns:
            List of certainty-related features.
        """
        return []

    def base_features(self, question, index, model):
        """
        Generates a list of base features for a given question.

        Args:
            question (tuple): A tuple containing the question text, category, and difficulty.
                - question[0] (str): The text of the question.
                - question[1] (str): The category of the question (e.g., "algebra").
                - question[2] (str): The difficulty level of the question.
            index: An index value (purpose not specified in the current context).
            model: The model used for feature generation (purpose not specified in the current context).

        Returns:
            list: A list of feature values including:
                - Normalized length of the question text.
                - Binary indicator for the "algebra" category.
                - Boolean indicators for various difficulty levels ("level 1" to "level 5", "easy", "medium", "hard").
        """
        features = [len(question[0]) / 1000]
        if question[1].lower() in ["algebra"]:
            features.append(1)
        else:
            features.append(0)
        for difficulty in [f"level {i}" for i in range(1, 6)] + ["easy", "medium", "hard"]:
            features.append(difficulty in question[2].lower())
        return features
