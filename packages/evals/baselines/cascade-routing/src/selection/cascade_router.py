from .base_algorithm import Algorithm
import numpy as np
from loguru import logger
from .lambda_strategy import ConstantStrategy

class CascadeRouter(Algorithm):
    def __init__(self, quality_computer, cost_computer, 
                 models, max_expected_cost, 
                 strategies=[ConstantStrategy(10000)], 
                 rounding_digits=8, greedy=False, 
                 force_order=True, max_depth=None, 
                 top_k_keep=None, set_sigma_none=False, 
                 cascade=False, do_speedup=True):
        """
        Initializes a CascadeRouter object.

        Args:
            quality_computer: The quality computer object used for computing the quality of models.
            cost_computer: The cost computer object used for computing the cost of models.
            models: A list of models to be considered for selection.
            max_expected_cost: The maximum expected cost allowed for selecting models.
            strategies: A list of hyperparameter search strategies to be used for model selection. 
                        Default is [ConstantStrategy(10000)].
            rounding_digits: The number of digits to round the computed values. Default is 8.
            greedy: A boolean indicating whether to use greedy selection. Default is False.
            force_order: A boolean indicating whether to force the execution of the models to be in the same order as the one given. 
                        Default is True.
            max_depth: The maximum depth allowed for supermodels in the model selection process. Default is None.
            top_k_keep: The number of top models to keep after each step in the selection. Reduces search time. 
                        Default is None.
            set_sigma_none: A boolean indicating whether to set the deviations of the computed quality estimates to None. 
                            Only used for ablation, should not be used in practice.
                            Default is False.
            cascade: A boolean indicating whether to use cascading instead of cascade routing. Default is False.
            do_speedup: A boolean indicating whether to perform speedup based on Lemma 1 in our paper. 
                        Only used for ablation, should not be used in practice.
                        Default is True.
        """
        super().__init__(quality_computer, cost_computer, models, 
                         max_expected_cost, strategies, rounding_digits)
        self.lambdas = None
        self.qualities = None
        self.costs = None
        self.gamma = None
        self.greedy = greedy
        self.force_order = force_order
        if max_depth is not None and max_depth > len(models):
            max_depth = None
        self.max_depth = max_depth
        self.top_k_keep = top_k_keep
        self.set_sigma_none = set_sigma_none
        self.cascade = cascade
        self.do_speedup = do_speedup
        if cascade:
            self.force_order = True

    def get_lambdas(self):
        """
        Returns the lambdas of the cascade router.

        :return: A list of lambdas.
        """
        return self.lambdas

    def predict(self, questions, model_answers):
        qualities, sigma_qualities = self.quality_computer.predict(questions, model_answers)
        costs = self.cost_computer.predict(questions, model_answers)
        # sum of the first i costs is cost of ith supermodel
        models = []
        none_lambdas  = sum([1 for lambda_ in self.lambdas if lambda_ is None])
        max_depth = len(self.lambdas) - none_lambdas
        if self.max_depth is not None:
            max_depth = min(self.max_depth, max_depth)
        for i in range(len(questions)):
            step = len([j for j in model_answers[i] if j is not None])
            if step >= max_depth:
                models.append(None)
                continue
            lambda_ = self.lambdas[step]
            model = self._predict_model(questions[i], qualities[i], sigma_qualities[i], costs[i], 
                                                            model_answers[i], step, lambda_, 
                                                            max_depth=max_depth)[0]
            model = self.models[model] if model is not None else None
            models.append(model)
        return models
        
    def _predict_model(self, question, qualities_question, sigma_qualities, 
                       costs, model_answers_question, step=0, 
                       lambda_=None, most_expensive=False, 
                       cheapest=False, max_depth=None):
        """
        Predicts the best model to run based on the given parameters.
        Args:
            question (any): The question to be answered.
            qualities_question (list): The estimated qualities of each model for the question.
            sigma_qualities (float): The deviations for the estimated qualities.
            costs (list): The costs of running each model.
            model_answers_question (list): The answers of each model to the question. None if the model has not been run.
            step (int, optional): The current step. Defaults to 0.
            lambda_ (float, optional): The lambda value. Defaults to None.
            most_expensive (bool, optional): Flag indicating whether to select the most expensive model among the most optimal models. 
                                            Defaults to False.
            cheapest (bool, optional): Flag indicating whether to select the cheapest model  among the most optimal models. 
                                        Defaults to False.
            max_depth (int, optional): The maximum depth. Defaults to None.
        Returns:
            tuple: A tuple containing the model to run next and the list of models to evaluate afterwards if following the same strategy.
        """
        if self.max_depth is not None:
            max_depth = self.max_depth if max_depth is None else min(self.max_depth, max_depth)

        if (max_depth is not None and step >= max_depth) or step >= len(self.models):
            return None, []
        if lambda_ is None:
            lambda_ = self.lambdas[step]

        if self.set_sigma_none:
            sigma_qualities = None
        models_already_run = [i for i in range(len(model_answers_question)) 
                              if model_answers_question[i] is not None]

        models_to_evaluate = [(models_already_run, [], 0, 0)]
        step = 0

        best_models = dict()

        while len(models_to_evaluate) > 0 and (not self.greedy or step <= 1):
            next_models_to_evaluate = []
            for run_models, not_run_models, quality_parent_supermodel, _ in models_to_evaluate:
                all_models = run_models + not_run_models
                if max_depth is not None and len(all_models) > max_depth:
                    continue
                if len(all_models) == 0:
                    cost_supermodel = 0
                    quality_supermodel = -10 ** 8 # basically negative infinity
                else:
                    cost_supermodel = np.sum(costs[all_models])
                    quality_supermodel, _ = self.quality_computer.predict_supermodels(
                        [question],
                        [all_models],
                        [qualities_question],
                        [sigma_qualities],
                        [model_answers_question]
                    )
                if len(not_run_models) > 0 and self.do_speedup and not self.cascade and len(all_models) > 1:
                    cost_last_model = costs[not_run_models[-1]]
                    if (quality_supermodel - quality_parent_supermodel - lambda_ * cost_last_model) < 0:
                        continue
                tradeoff = np.round(quality_supermodel - lambda_ * cost_supermodel, self.rounding_digits)
                if len(all_models) > 0 and ('all' not in best_models or tradeoff > best_models['all'][0][2]):
                    best_models['all'] = [(run_models, not_run_models, tradeoff, cost_supermodel)]
                elif len(all_models) > 0 and tradeoff == best_models['all'][0][2]:
                    best_models['all'] += [(run_models, not_run_models, tradeoff, cost_supermodel)]

                models_possibilities = [i for i in range(len(model_answers_question)) 
                                        if i not in all_models]
                if self.force_order and len(all_models) > 0:
                    models_possibilities = [i for i in models_possibilities if i > all_models[-1]]
                if len(not_run_models) > 0:
                    models_possibilities = [i for i in models_possibilities if i > not_run_models[-1]] # prevent duplicates
                if self.cascade:
                    if len(all_models) == 0:
                        models_possibilities = [0]
                    elif all_models[-1] < len(model_answers_question) - 1:
                        models_possibilities = [all_models[-1] + 1]
                
                for model in models_possibilities:
                    not_run_models_new = not_run_models + [model]
                    next_models_to_evaluate.append((run_models, not_run_models_new, 
                                                    quality_supermodel, tradeoff))
                
                step += 1

                if self.top_k_keep is not None:
                    next_models_to_evaluate = sorted(next_models_to_evaluate, key=lambda x: x[3], reverse=True)[:self.top_k_keep]

            models_to_evaluate = next_models_to_evaluate[:]
        
        if cheapest or (not most_expensive and np.random.uniform() >= self.gamma):
            best_index = np.argmin([best_models['all'][i][3] for i in range(len(best_models['all']))])
        else:
            best_index = np.argmax([best_models['all'][i][3] for i in range(len(best_models['all']))])
        supermodel = best_models['all'][best_index]

        if len(supermodel[1]) == 0:
            return None, []
        
        if not self.force_order:
            index_model_to_run = np.argmin(best_models[model][2] for model in supermodel[1])
        else:
            index_model_to_run = 0
        model_to_run = supermodel[1][index_model_to_run]
        return model_to_run, supermodel[1]

        
    def fit(self, questions, model_answers, ground_truth_qualities=None, ground_truth_costs=None):
        self.quality_computer.trigger_training(True)
        self.cost_computer.trigger_training(True)
        self.lambdas = [0 for _ in range(len(self.models))]
        if self.max_depth is not None:
            self.lambdas = [0 for _ in range(self.max_depth)]
        current_quality = -np.inf

        for strategy in self.strategies:
            lambdas, cost, quality = strategy.compute_lambdas(self.lambdas, 
                                                              self._execute, 
                                                              self.max_expected_cost, 
                                                              (questions, 
                                                               model_answers, 
                                                               ground_truth_qualities, 
                                                               ground_truth_costs))
            if quality is not None and cost is not None and quality > current_quality and \
                (cost <= self.max_expected_cost or (current_quality == -np.inf and all([lambda_ > strategy.max_lambda for lambda_ in lambdas]))):
                self.lambdas = lambdas
                current_quality = quality

        quality_cheap,cost_cheap,quality_expensive,cost_expensive = self._execute_cheap_expensive(self.lambdas, 
                                                                                                  questions, 
                                                                                                  model_answers,
                                                                                                  ground_truth_qualities, 
                                                                                                  ground_truth_costs)
        
        if cost_expensive == cost_cheap:
            self.gamma = 0
        else:
            self.gamma = (self.max_expected_cost - cost_cheap) / (cost_expensive - cost_cheap)
            self.gamma = np.clip(self.gamma, 0, 1)
        
        logger.info(f"Actual Final Lambdas: {self.lambdas}")
        logger.info(f"Actual Final Cost: {(1 - self.gamma) * cost_cheap + self.gamma * cost_expensive}")
        logger.info(f"Actual Final Quality: {(1 - self.gamma) * quality_cheap + self.gamma * quality_expensive}")

        self.quality_computer.trigger_training(False)
        self.cost_computer.trigger_training(False)

    def select_answer(self, questions, model_answers):
        models_selected = []
        qualities, sigma_qualities = self.quality_computer.predict(questions, model_answers)
        for i, quality in enumerate(qualities):
            indices_with_answer = [j for j in range(len(quality)) if model_answers[i][j] is not None]
            if len(indices_with_answer) == 0:
                models_selected.append(None)
            models_selected.append(self.models[indices_with_answer[np.argmax(quality[indices_with_answer])]])
        return models_selected

    def _execute_cheap_expensive(self, lambdas, questions, model_answers, ground_truth_qualities, ground_truth_costs):
        output_dict_cheap = self._execute(lambdas, questions, model_answers,
                                        cheapest=True, most_expensive=False,
                                        ground_truth_qualities=ground_truth_qualities,
                                        ground_truth_costs=ground_truth_costs)
        quality_cheap = output_dict_cheap['quality']
        cost_cheap = output_dict_cheap['cost']
        output_dict_expensive = self._execute(lambdas, questions, model_answers,
                                            cheapest=False, most_expensive=True,
                                            ground_truth_qualities=ground_truth_qualities,
                                            ground_truth_costs=ground_truth_costs)
        quality_expensive = output_dict_expensive['quality']
        cost_expensive = output_dict_expensive['cost']
        return quality_cheap, cost_cheap, quality_expensive, cost_expensive

    def _execute(self, lambdas, questions, model_answers,
                 ground_truth_qualities=None, ground_truth_costs=None,
                cheapest=True, most_expensive=False):
        """
        Executes the cascade router algorithm to select models based on given parameters.
        Args:
            lambdas (list): List of lambda values for each step in the cascade router.
            questions (list): List of questions to be answered by the models.
            model_answers (list): List of model answers for each question.
            ground_truth_qualities (list, optional): List of ground truth qualities for each model answer. Defaults to None.
            ground_truth_costs (list, optional): List of ground truth costs for each model answer. Defaults to None.
            cheapest (bool, optional): Flag indicating whether to select the cheapest model. Defaults to True.
            most_expensive (bool, optional): Flag indicating whether to select the most expensive model. Defaults to False.
        Returns:
            dict: Dictionary containing the average cost and quality of the selected models.
        """
        cost = 0
        quality = 0
        done = [False for _ in range(len(questions))]
        models_run = [[] for _ in range(len(questions))]
        none_lambdas = sum([1 for lambda_ in lambdas if lambda_ is None])
        max_depth = len(lambdas) - none_lambdas
        for step in range(len(self.models) + 1):
            end_index = step
            for i in range(len(questions)):
                if done[i]:
                    continue
                model_answers_sample = [model_answers[i][j] if j in models_run[i][:end_index] else None 
                                        for j in range(len(self.models))]
                qualities, sigma_qualities = self.quality_computer.predict([questions[i]], 
                                                                           [model_answers_sample])
                qualities = qualities[0]
                sigma_qualities = sigma_qualities[0]
                costs = self.cost_computer.predict([questions[i]], [model_answers_sample])
                costs = costs[0]
                if len(lambdas) > step:
                    lambda_ = lambdas[step]
                    model_here, future_models = self._predict_model(questions[i], 
                                                                    qualities, sigma_qualities, 
                                                                    costs, model_answers_sample,
                                                                    step, lambda_, 
                                                                    cheapest=cheapest, 
                                                                    most_expensive=most_expensive, 
                                                                    max_depth=max_depth)
                else:
                    model_here, future_models = None, []
                

                if model_here is None:
                    models_run_here = models_run[i] + future_models
                    
                    selected_model = models_run_here[np.argmax(qualities[models_run_here])]
                    if ground_truth_qualities is not None:
                        quality += ground_truth_qualities[i][selected_model]
                    else:
                        quality += qualities[selected_model]
                    if ground_truth_costs is not None:
                        cost += np.sum(ground_truth_costs[i][models_run_here])
                    else:
                        cost += np.sum(costs[models_run_here])
                    done[i] = True
                else:
                    models_run[i].append(model_here)

            if all(done):
                break

        output_dict = {
            'cost': cost / len(questions),
            'quality': quality / len(questions),
        }
        return output_dict

    