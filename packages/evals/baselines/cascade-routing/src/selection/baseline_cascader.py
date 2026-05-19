import numpy as np
from loguru import logger
from .base_algorithm import Algorithm
from .lambda_strategy import ConstantStrategy

class BaselineCascader(Algorithm):
    def __init__(self, quality_computer, cost_computer, 
                 models, max_expected_cost, 
                 strategies=[ConstantStrategy(1)]):
        """
        Initialize the BaselineCascader object. This object implements the baseline cascader,
        which corresponds to the thresholding strategy for model selection in our paper.

        Args:
            quality_computer: The quality computer object used for computing the quality of models.
            cost_computer: The cost computer object used for computing the cost of models.
            models: A list of models to be considered for selection.
            max_expected_cost: The maximum expected cost allowed for model selection.
            strategies: A list of hyperparameter search strategies to be used for model selection. 
                        Default is [ConstantStrategy(1)].
        """
        super().__init__(quality_computer, cost_computer, 
                         models, max_expected_cost, 
                         strategies, 0)
        self.lambdas = None

    def get_lambdas(self):
        """
        Returns the lambdas used in the baseline cascader.

        Returns:
            list: The lambdas used in the baseline cascader.
        """
        return self.lambdas

    def predict(self, questions, model_answers):
        qualities, _ = self.quality_computer.predict(questions, model_answers)
        models = []
        for i in range(len(questions)):
            step = len([answer for answer in model_answers[i] if answer is not None])
            if step == 0:
                models.append(self.models[0])
                continue
            run_next = self._predict_model(qualities[i], step)
            models.append(self.models[step] if run_next > 0 else None)
        return models

    def select_answer(self, questions, model_answers):
        models_selected = []
        qualities, _ = self.quality_computer.predict(questions, model_answers)
        for i, quality in enumerate(qualities):
            indices_with_answer = [j for j in range(len(quality)) if model_answers[i][j] is not None]
            if len(indices_with_answer) == 0:
                models_selected.append(None)
            else:
                models_selected.append(self.models[np.max(indices_with_answer)])

        return models_selected
    
    def _predict_model(self, qualities_question, step=0, lambda_=None):
        """
        Predicts the model based on the given qualities_question, step, and lambda_.

        Parameters:
        - qualities_question (list): A list of qualities for the question.
        - step (int): The current step in the model selection process.
        - lambda_ (float): The lambda value used for the prediction. 
                            If not provided, it uses the lambda value corresponding to the current step.

        Returns:
        - int: Whether or not to continue to the next step.
        """
        if step == 0:
            return 1
        if lambda_ is None:
            lambda_ = self.lambdas[step - 1]
        if qualities_question[step - 1] < 1 - lambda_:
            return 1
        else:
            return 0
        
    def fit(self, questions, model_answers, 
            ground_truth_qualities=None, ground_truth_costs=None):
        self.quality_computer.trigger_training(True)
        self.cost_computer.trigger_training(True)
        self.lambdas = [0 for _ in range(len(self.models) - 1)]
        current_quality = -np.inf
        qualities_per_step, costs_per_step = self.generate_step_data(questions, model_answers)
        print(qualities_per_step)
        if ground_truth_qualities is None:
            ground_truth_qualities = qualities_per_step[-1]
        if ground_truth_costs is None:
            ground_truth_costs = costs_per_step[-1]

        for strategy in self.strategies:
            new_lambdas, cost, quality = strategy.compute_lambdas(self.lambdas, 
                                                                  self._execute, 
                                                                  self.max_expected_cost,
                                                                  (qualities_per_step, 
                                                                   ground_truth_qualities, 
                                                                   ground_truth_costs))
            
            if quality is not None and cost is not None and quality > current_quality and \
                (cost <= self.max_expected_cost or (current_quality == -np.inf and all([lambda_ > strategy.max_lambda for lambda_ in new_lambdas]))):
                self.lambdas = new_lambdas
                current_quality = quality
            elif cost is not None and cost > self.max_expected_cost:
                logger.info(f"Cost {cost} is higher than maximum expected cost {self.max_expected_cost}. Stopping.")
        
        self.quality_computer.trigger_training(False)
        self.cost_computer.trigger_training(False)

    def generate_step_data(self, questions, model_answers):
        """
        Generates step data for the baseline cascader. This allows us to iterate quicker
        in the hyperparameter optimization process.
        Args:
            questions (list): A list of questions.
            model_answers (list): A list of model answers.
        Returns:
            tuple: A tuple containing two lists - qualities_per_step and costs_per_step.
                - qualities_per_step (list): A list of qualities for each step.
                - costs_per_step (list): A list of costs for each step.
        """
        qualities_per_step = []
        costs_per_step = []
        
        for step in range(1, len(self.models)):
            model_answers_step = [
                list(model_answers[i][:step]) + [None for _ in range(len(model_answers[i]) - step)] 
                    if model_answers[i] is not None else None 
                for i in range(len(model_answers))
            ]
            qualities, _ = self.quality_computer.predict(questions, model_answers_step)
            qualities_per_step.append(qualities)
            costs = self.cost_computer.predict(questions, model_answers_step)
            costs_per_step.append(costs)
        return qualities_per_step, costs_per_step

    def _execute(self, lambdas, qualities_step,
                ground_truth_qualities, ground_truth_costs):
        """
        Executes the baseline cascader algorithm on the training data.

        Args:
            lambdas (list): List of lambda values for each step.
            qualities_step (list): List of qualities for each step.
            ground_truth_qualities (list): List of ground truth qualities.
            ground_truth_costs (list): List of ground truth costs.

        Returns:
            dict: A dictionary containing the cost and quality values.
        """
        cost = 0
        quality = 0
        done = [False for _ in range(len(ground_truth_qualities))]
        for step in range(1, len(self.models)):
            step_index = step - 1
            qualities = qualities_step[step_index]
            lambda_ = lambdas[step_index]
            for i in range(len(ground_truth_qualities)):
                if done[i]:
                    continue
                continue_here = self._predict_model(qualities[i], step, lambda_)
                if continue_here == 0 or step == len(self.models) - 1:
                    quality += ground_truth_qualities[i][step + continue_here - 1]
                    cost += np.sum(ground_truth_costs[i][:step + continue_here])
                    done[i] = True
            if all(done):
                break
        return {
            'cost': cost / len(ground_truth_qualities), 
            'quality': quality / len(ground_truth_qualities),
        }
        