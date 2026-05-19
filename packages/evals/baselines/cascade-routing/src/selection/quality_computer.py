import numpy as np

from .base_computer import BaseComputer
from .statistics import compute_expected_max
from itertools import combinations

from sklearn.linear_model import LinearRegression

class BaseQualityComputer(BaseComputer):
    def __init__(self, n_samples=100):
        """
        Initialize the QualityComputer object.

        Parameters:
            n_samples (int): The number of samples to be used for max quality computation of supermodels. 
                            Default is 100.
        """
        self.n_samples = n_samples
        self.covariances_default = None
        super().__init__()
    
    def predict_supermodels(
            self, 
            questions,
            indices_models_supermodel,
            qualities,
            sigma_qualities,
            model_answers
    ):
        """
        Predicts the qualities of supermodels based on the given inputs.
        Args:
            questions (list): A list of questions.
            indices_models_supermodel (list): A list of indices indicating the supermodels to consider for each question.
            qualities (list): A list of qualities for each question and model.
            sigma_qualities (list): A list of covariance matrices for the qualities of each question and supermodel.
            model_answers (list): A list of model answers.
        Returns:
            qualities_supermodel (ndarray): An array of predicted qualities for each supermodel.
            qualities_var_supermodel (ndarray): An array of variances for the predicted qualities of each supermodel.
        """
        qualities_supermodel = []
        qualities_var_supermodel = []
        for i, question in enumerate(questions):
            qualities_sample = qualities[i][indices_models_supermodel[i]]
            if sigma_qualities[i] is None:
                sigma_qualities_sample = None
            else:
                sigma_qualities_sample = sigma_qualities[i][np.ix_(indices_models_supermodel[i], indices_models_supermodel[i])]
            
            qual, var = compute_expected_max(
                    qualities_sample,
                    sigma_qualities_sample,
                    independent=self.is_independent,
                    n_samples=self.n_samples
                )
            qualities_supermodel.append(qual)
            qualities_var_supermodel.append(var)
        return np.array(qualities_supermodel), np.array(qualities_var_supermodel)
    
    def fit_covariances(self, questions, model_answers):
        """
        Computes and stores the covariance matrices for different combinations of models.

        This method calculates the covariance matrices for the predictions of different 
        combinations of models and stores them in the `covariances_default` attribute.

        This is a default strategy that you can use to compute the covariance matrices. 
        Note that this assumes that your "predict" method is already implemented.

        Args:
            questions (list): A list of questions or input data for which predictions are made.
            model_answers (list of lists): A list where each element is a list of answers 
                                        from different models. Assumes no `None` values 
                                        in `model_answers`.

        Notes:
            - The method assumes that `model_answers` contains no `None` values.
            - The `predict` method is used to generate predictions for the given questions 
            and model answers.
            - The covariance matrices are computed for all possible combinations of models 
            (including the empty set) and stored in a dictionary with keys representing 
            the combination of models.
        """
        self.covariances_default = dict()
        n_models = len(model_answers[0])
        predictions_all_models = self.predict(
            questions, model_answers
        )[0]
        for n_models_computed in range(n_models + 1):
            for models_computed in combinations(range(n_models), n_models_computed):
                str_rep = ','.join([str(model) for model in sorted(models_computed)])
                model_answers_here = [
                    [answers[i] if i in models_computed else None for i in range(n_models)]
                    for answers in model_answers
                ]
                predictions = self.predict(
                    questions, model_answers_here
                )[0]
                cov = np.cov((np.array(predictions) - np.array(predictions_all_models)).T)
                self.covariances_default[str_rep] = cov

    def predict_covariances(self, questions, model_answers):
        """
        Predicts the covariances for a given set of questions and model answers using the default strategies.
        Args:
            questions (list): A list of questions for which covariances are to be predicted.
            model_answers (list of list): A list where each element is a list of answers from different models.
        Returns:
            list: A list of predicted covariances corresponding to each question. If `self.covariances_default` is None,
              returns a list of None values with the same length as `questions`.
        """
        if self.covariances_default is None:
            return [None] * len(questions)
        
        covariances = []
        for answers in model_answers:
            not_none_indices = [i for i, ans in enumerate(answers) if ans is not None]
            str_rep = ','.join([str(i) for i in sorted(not_none_indices)])
            if str_rep not in self.covariances_default:
                covariances.append(None)
            else:
                covariances.append(self.covariances_default[str_rep])
        return covariances

class GroundTruthQualityComputer(BaseQualityComputer):
    def __init__(self, noise_before_run=0.2, noise_after_run=0.05, n_samples=100):
        """
        Initializes the GroundTruthQualityComputer object.
        Computes the quality by adding noise to the ground truth quality values and 
        then fitting a linear model to the noisy values.

        Args:
            noise_before_run (float): The amount of noise before running the computation. Defaults to 0.2.
            noise_after_run (float): The amount of noise after running the computation. Defaults to 0.05.
            n_samples (int): The number of samples. Defaults to 100.
        """
        super().__init__(n_samples)
        self.noise_before_run = noise_before_run
        self.noise_after_run = noise_after_run
        self.quality_mapping = None
        self.sigmas = None

    def fit(self, questions, model_answers, measure):
        self.quality_mapping = dict()
        noisy_values = []
        for measure_value in measure:
            noisy_value = []
            for i in range(len(measure_value)):
                val = measure_value[i]
                noisy_value.append([
                    np.random.normal(val, self.noise_before_run),
                    np.random.normal(val, self.noise_after_run)
                ])
            noisy_value = np.array(noisy_value)
            noisy_values.append(noisy_value)

        noisy_values = np.array(noisy_values)

        actual_values = np.zeros(noisy_values.shape)

        self.sigmas = [0 for _ in range(measure.shape[1])]

        for model in range(noisy_values.shape[1]):
            for i in range(noisy_values.shape[2]):
                linear_model = LinearRegression()
                linear_model.fit(noisy_values[:, model, i].reshape(-1, 1), measure[:, model])
                actual_values[:, model, i] = linear_model.predict(noisy_values[:, model, i].reshape(-1, 1))
            
            self.sigmas[model] = np.std(actual_values[:, model, 0] - actual_values[:, model, 1])

        for q, a in zip(questions, actual_values):
            self.quality_mapping[q] = a

    def predict(self, questions, model_answers):
        qualities = []
        sigma_qualities = []
        for question, model_answer in zip(questions, model_answers):
            value = self.quality_mapping[question]
            value = np.array([
                value[i][0] if answer is None else value[i][1] for i, answer in enumerate(model_answer)
            ])
            sigma_noise = np.diag([
                self.sigmas[i] ** 2 if answer is None else 1e-6
                for i, answer in enumerate(model_answer)
            ])
            qualities.append(value)
            sigma_qualities.append(sigma_noise)
        
        return np.array(qualities), np.array(sigma_qualities)