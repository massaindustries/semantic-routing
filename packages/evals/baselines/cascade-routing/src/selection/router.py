import numpy as np
from .base_algorithm import Algorithm
from .lambda_strategy import ConstantStrategy
from collections import Counter

class Router(Algorithm):
    def __init__(self, quality_computer, cost_computer, models, max_expected_cost,
                 rounding_digits=8, strategies=[ConstantStrategy(10000, 50)]):
        """
        Initialize the Router object.

        Args:
            quality_computer: The quality computer object used for computing model qualities.
            cost_computer: The cost computer object used for computing model costs.
            models: A list of model objects.
            max_expected_cost: The maximum expected cost.
            rounding_digits: The number of digits to round the results to (default: 8).
            strategies: A list of strategy objects for hyperparameter optimization 
                        (default: [ConstantStrategy(10000, 50)]).
        """
        super().__init__(quality_computer, cost_computer, models, 
                         max_expected_cost, strategies, rounding_digits)
        self.lambda_ = None
        self.qualities = None
        self.costs = None

    def compute_budget(self, qualities, costs, ground_truth_qualities=None, ground_truth_costs=None):
        """
        Computes the optimal lambdas for model selection based on the given qualities and costs.
        Args:
            qualities (list): List of qualities for each model.
            costs (list): List of costs for each model.
            ground_truth_qualities (list, optional): List of ground truth qualities for each model. 
                                                    Defaults to None.
            ground_truth_costs (list, optional): List of ground truth costs for each model. 
                                                Defaults to None.
        Returns:
            None
        """
        best_lambda = None
        best_quality = None
        for strategy in self.strategies:
            lambdas, cost, quality = strategy.compute_lambdas([0], self._execute, self.max_expected_cost, 
                                      (qualities, costs, ground_truth_qualities, ground_truth_costs))
            if best_lambda is None or cost < self.max_expected_cost and (best_lambda is None or quality > best_quality):
                best_lambda = lambdas[0]
                best_quality = quality
            
        self.lambda_ = best_lambda
        output_cheap = self._execute(self.lambda_, qualities, costs, ground_truth_qualities, 
                                     ground_truth_costs, cheapest=True, most_expensive=False)
        output_expensive = self._execute(self.lambda_, qualities, costs, ground_truth_qualities, 
                                     ground_truth_costs, cheapest=False, most_expensive=True)
        cost_cheap = output_cheap['cost']
        cost_expensive = output_expensive['cost']
        if cost_cheap == cost_expensive:
            self.gamma = 1
        else:
            self.gamma = (cost_expensive - self.max_expected_cost) / (cost_expensive - cost_cheap)
            self.gamma = min(1, max(0, self.gamma))

    def get_lambdas(self):
        return [self.lambda_]

    def fit(self, questions, model_answers, ground_truth_qualities=None, ground_truth_costs=None):
        self.quality_computer.trigger_training(True)
        self.cost_computer.trigger_training(True)
        qualities, _ = self.quality_computer.predict(questions, [[None] * len(self.models)] * len(questions))
        costs = self.cost_computer.predict(questions, [[None] * len(self.models)] * len(questions))

        self.compute_budget(qualities, costs, ground_truth_qualities, ground_truth_costs)
        self.quality_computer.trigger_training(False)
        self.cost_computer.trigger_training(False)
    
    def predict(self, questions, model_answers=None):
        qualities_question, _ = self.quality_computer.predict(questions, model_answers)
        costs_question = self.cost_computer.predict(questions, model_answers)
        return [self._predict_model(qualities_question[i], costs_question[i], cheapest=False, most_expensive=False) for i in range(len(questions))]
    
    def select_answer(self, questions, model_answers):
        models_selected = []
        for i, model_answer in enumerate(model_answers):
            indices_with_answer = [j for j in range(len(model_answer)) if model_answer[j] is not None]
            if len(indices_with_answer) == 0:
                models_selected.append(None)
            else:
                models_selected.append(self.models[indices_with_answer[0]])
        return models_selected
        
    def _predict_model(self, qualities_question, costs_model, 
                       lambda_=None, cheapest=True, most_expensive=False):
        """
        Predicts the best model based on the qualities of the question and the costs of the models.
        Args:
            qualities_question (numpy.ndarray): An array representing the qualities of the question.
            costs_model (numpy.ndarray): An array representing the costs of the models.
            lambda_ (float, optional): A parameter used to adjust the importance of qualities and costs. If not provided, the default value from the class instance will be used.
            cheapest (bool, optional): If True, the cheapest model among the best models will be selected. Default is True.
            most_expensive (bool, optional): If True, the most expensive model among the best models will be selected. Default is False.
        Returns:
            best_model: The best model based on the given qualities and costs.
        """
        if lambda_ is None:
            lambda_ = self.lambda_
    
        highest = np.round(qualities_question - lambda_ * costs_model, self.rounding_digits)
        # check which model has the highest value. If multiple, return the cheapest
        max_value = np.max(highest)
        
        # collect all models with the highest value
        best_models = np.where(highest == max_value)[0]
        # select the cheapest model
        if len(best_models) > 1:
            best_model_cheapest = best_models[np.argmin(costs_model[best_models])]
            best_model_expensive = best_models[np.argmax(costs_model[best_models])]
            if cheapest:
                best_model = best_model_cheapest
            elif most_expensive:
                best_model = best_model_expensive
            elif np.random.rand() < self.gamma:
                best_model = best_model_expensive
            else:
                best_model = best_model_cheapest
        else:
            best_model = best_models[0]
        best_model = self.models[best_model]
        return best_model
    
    def _execute(self, lambda_, qualities, costs, ground_truth_qualities=None, ground_truth_costs=None, 
                 cheapest=True, most_expensive=False):
        """
        Executes the model selection process based on the given parameters.

        Args:
            lambda_ (float or list): The regularization parameter(s) for the model selection.
            qualities (list): The list of quality values for each model.
            costs (list): The list of cost values for each model.
            ground_truth_qualities (list, optional): The list of ground truth quality values for each model. Default is None.
            ground_truth_costs (list, optional): The list of ground truth cost values for each model. Default is None.
            cheapest (bool, optional): Flag indicating whether to select the cheapest model. Default is True.
            most_expensive (bool, optional): Flag indicating whether to select the most expensive model. Default is False.

        Returns:
        - dict: A dictionary containing the following keys:
            - 'cost': The mean cost value of the selected models.
            - 'quality': The mean quality value of the selected models.
            - 'models_run': A Counter object containing the count of each model selected.

        """
        if isinstance(lambda_, list):
            lambda_ = lambda_[0]
        all_costs = []
        all_qualities = []
        models_run = []
        for i in range(len(qualities)):
            best_model = self._predict_model(qualities[i], costs[i], lambda_, cheapest, most_expensive)
            best_model_index = self.models.index(best_model)
            models_run.append(best_model_index)
            if ground_truth_costs is not None:
                all_costs.append(ground_truth_costs[i][best_model_index])
            else:
                all_costs.append(costs[i][best_model_index])
            if ground_truth_qualities is not None:
                all_qualities.append(ground_truth_qualities[i][best_model_index])
            else:
                all_qualities.append(qualities[i][best_model_index])

        return {
            'cost': np.mean(all_costs),
            'quality': np.mean(all_qualities),
            'models_run': Counter(models_run)
        }
