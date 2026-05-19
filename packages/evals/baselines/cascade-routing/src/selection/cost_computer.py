import numpy as np
from .base_computer import BaseComputer

from sklearn.linear_model import LinearRegression

class BaseCostComputer(BaseComputer):
    def predict(self, questions, model_answers):
        """
        Predict the cost of the given model answers.

        Args:
            questions (list): List of questions.
            model_answers (list): List of model answers.

        Returns:
            list: A list of predictions. Each question should have a corresponding prediction for each model.
        """
        raise NotImplementedError
    
class GroundTruthCostComputer(BaseCostComputer):
    def __init__(self, noise_before_run, noise_after_run, assume_constant=False):
        """
        Initialize the CostComputer object.
        Computes the cost by adding noise to the ground truth cost values and then fitting a linear model 
        to the noisy values.

        Args:
            noise_before_run (float): The noise value before running model computation.
            noise_after_run (float): The noise value after running model computation.
            assume_constant (bool, optional): Flag indicating whether to set computed cost to a constant for each model. 
                                                Defaults to False.
        """
        super().__init__()
        self.noise_before_run = noise_before_run
        self.noise_after_run = noise_after_run
        self.assume_constant = assume_constant
        self.cost_mapping = None

    def fit(self, questions, model_answers, measure):
        self.cost_mapping = dict()
        noisy_values = []
        for measure_value in measure:
            value = [
                [float(measure_value[i] + np.random.normal(0, self.noise_before_run)), 
                 float(measure_value[i] + np.random.normal(0, self.noise_after_run))] 
                 for i in range(len(measure_value))
            ]
            noisy_values.append(value)

        if self.assume_constant:
            self.average_costs = np.mean(measure, axis=0)

        noisy_values = np.array(noisy_values)

        actual_values = np.zeros(noisy_values.shape)

        for model in range(noisy_values.shape[1]):
            for i in range(noisy_values.shape[2]):
                linear_model = LinearRegression()
                linear_model.fit(noisy_values[:, model, i].reshape(-1, 1), measure[:, model])
                actual_values[:, model, i] = linear_model.predict(noisy_values[:, model, i].reshape(-1, 1))
        
        for q, a in zip(questions, actual_values):
            self.cost_mapping[q] = a



    def predict(self, questions, model_answers):
        qualities = []
        for question, model_answer in zip(questions, model_answers):
            if not self.assume_constant:
                value = self.cost_mapping[question]
                value = np.array([
                    value[i][0] if answer is None else value[i][1] for i, answer in enumerate(model_answer)
                ])
            else:
                value = self.average_costs
            qualities.append(value)

        return np.array(qualities)