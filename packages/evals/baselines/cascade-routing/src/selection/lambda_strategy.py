import numpy as np
from loguru import logger
from hyperopt import fmin, tpe, hp


class Strategy:
    def __init__(self):
        """
        Initializes an instance of the Strategy class. These strategies optimize the lambda parameters.
        """

        pass

    def compute_lambdas(self, current_optimal_lambdas, execute_function, 
                        cost_target, args_execute_function):
        """
        Compute the lambdas for model selection.

        Parameters:
        - current_optimal_lambdas (list): The current optimal lambdas in the selection algorithm.
        - execute_function (function): The function to execute for each lambda which returns both cost and quality.
        - cost_target (float): The target cost for model selection.
        - args_execute_function (tuple): The arguments to pass to the execute_function.

        Raises:
        - NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError
    
class ConstantStrategy(Strategy):
    def __init__(self, max_lambda, n_iterations=20):
        """
        Initialize a ConstantStrategy object.
        Assumes that all values of lambda are the same and performs a binary search to find the optimal lambda.

        Parameters:
        - max_lambda (float): The maximum value for lambda.
        - n_iterations (int, optional): The number of iterations. Default is 20.
        """
        super().__init__()
        self.max_lambda = max_lambda
        self.n_iterations = n_iterations

    def compute_lambdas(self, current_optimal_lambdas, execute_function, cost_target, args_execute_function):
        logger.debug("Computing lambdas with constant strategy")
        lambda_min = (0, None, None)
        lambda_max = (self.max_lambda, None, None)

        max_lambda = self.max_lambda
        while max_lambda < 1e10:
            lambdas = [max_lambda for _ in range(len(current_optimal_lambdas))]
            output_dict = execute_function(lambdas, *args_execute_function)
            cost = output_dict['cost']
            if cost > cost_target:
                logger.info(f'Max lambda {max_lambda} is too low, increasing max lambda')
                lambda_min = (max_lambda, cost, output_dict['quality'])
                max_lambda *= 2
                lambda_max = (max_lambda, None, None)
            else:
                lambda_max = (max_lambda, cost, output_dict['quality'])
                break

        # Do binary search to find the optimal lambda
        for _ in range(self.n_iterations):
            lambda_mid = (lambda_min[0] + lambda_max[0]) / 2
            logger.debug(f"Lambda mid: {lambda_mid}")
            lambdas = [lambda_mid for _ in range(len(current_optimal_lambdas))]
            output_dict = execute_function(lambdas, *args_execute_function)
            cost = output_dict['cost']

            if cost < cost_target:
                lambda_max = (lambda_mid, cost, output_dict['quality'])
            else:
                lambda_min = (lambda_mid, cost, output_dict['quality'])
            logger.debug(f"Cost: {cost}")
            logger.debug(f"Quality: {output_dict['quality']}")

        if lambda_max[2] is None:
            lambdas = [lambda_max[0] for _ in range(len(current_optimal_lambdas))]
            output_dict = execute_function(lambdas, *args_execute_function)
            lambda_max = (lambda_max[0], output_dict['cost'], output_dict['quality'])

        logger.info(f"Final Lambda: {lambda_max[0]}")
        logger.info(f"Final Cost: {lambda_max[1]}")
        logger.info(f"Final Quality: {lambda_max[2]}")
        
        return [lambda_max[0] for _ in range(len(current_optimal_lambdas))], lambda_max[1], lambda_max[2]
    
class RepetitiveConstantStrategy(Strategy):
    def __init__(self, max_lambda, n_iterations=20):
        """
        Initialize the RepetitiveConstantStrategy object.
        Assumes that lambda values are of the form [lambda, ..., lambda, None, ..., None] 
        and performs a binary search to find the optimal lambda and number of None values.

        Parameters:
        - max_lambda (float): The maximum value for lambda.
        - n_iterations (int, optional): The number of iterations. Default is 20.
        """
        super().__init__()
        self.max_lambda = max_lambda
        self.n_iterations = n_iterations

    def compute_lambdas(self, current_optimal_lambdas, execute_function, cost_target, args_execute_function):
        logger.debug("Computing lambdas with repetitive constant strategy")
        
        optimal_lambdas = None
        optimal_value = None
        optimal_cost = None
        # Do binary search to find the optimal lambda
        for i in range(len(current_optimal_lambdas)):
            lambda_min = (0, None, None)
            lambda_max = (self.max_lambda, None, None)

            max_lambda = self.max_lambda
            while max_lambda < 1e10:
                lambdas = [max_lambda for _ in range(len(current_optimal_lambdas))]
                output_dict = execute_function(lambdas, *args_execute_function)
                cost = output_dict['cost']
                if cost > cost_target:
                    logger.info(f'Max lambda {max_lambda} is too low, increasing max lambda')
                    lambda_min = (max_lambda, cost, output_dict['quality'])
                    max_lambda *= 2
                    lambda_max = (max_lambda, None, None)
                else:
                    lambda_max = (max_lambda, cost, output_dict['quality'])
                    break
            logger.debug(f"Step: {i}")
            for _ in range(self.n_iterations):
                lambda_mid = (lambda_min[0] + lambda_max[0]) / 2
                logger.debug(f"Lambda mid: {lambda_mid}")
                lambdas = [lambda_mid if j <= i else None for j in range(len(current_optimal_lambdas))]
                output_dict = execute_function(lambdas, *args_execute_function)
                cost = output_dict['cost']

                if cost < cost_target:
                    lambda_max = (lambda_mid, cost, output_dict['quality'])
                else:
                    lambda_min = (lambda_mid, cost, output_dict['quality'])
                logger.debug(f"Cost: {cost}")
                logger.debug(f"Quality: {output_dict['quality']}")

            if optimal_value is None or (lambda_max[2] is not None and lambda_max[2] > optimal_value):
                optimal_lambdas = [lambda_max[0] if j <= i else None for j in range(len(current_optimal_lambdas))]
                optimal_value = lambda_max[2]
                optimal_cost = lambda_max[1]

        logger.info(f"Final Lambda: {optimal_lambdas}")
        logger.info(f"Final Cost: {optimal_cost}")
        logger.info(f"Final Quality: {optimal_value}")
            
        return optimal_lambdas, optimal_cost, optimal_value

class HyperoptStrategy(Strategy):
    def __init__(self, max_lambda, n_searches=100, max_factor=4, from_scratch=False, 
                optimize_max_depth=False):
        """
        Initialize the HyperoptStrategy object.
        Uses the hyperopt library to optimize the lambda values.

        Parameters:
        - max_lambda (int): The maximum lambda value.
        - n_searches (int, optional): The number of searches to perform. Defaults to 100.
        - max_factor (int, optional): The maximum factor to increase the prior optimal lambdas by. Defaults to 4.
        - from_scratch (bool, optional): Whether to start from scratch and ignore the prior optimal lambdas. 
                                        Defaults to False.
        - optimize_max_depth (bool, optional): Whether to optimize the maximum depth. Defaults to False.
        """
        super().__init__()
        self.max_lambda = max_lambda
        self.n_searches = n_searches
        self.max_factor = max_factor
        self.from_scratch = from_scratch
        self.optimize_max_depth = optimize_max_depth
        self.all_results = []

    def objective(self, lambdas, cost_init, lambda_tradeoff, execute_function, *args):
        if self.optimize_max_depth:
            lambdas, max_depth = lambdas[:-1], int(lambdas[-1])
            lambdas = [lambda_ if i < max_depth else None for i, lambda_ in enumerate(lambdas)]
        output_dict = execute_function(lambdas, *args)
        output_dict['lambdas'] = lambdas
        self.all_results.append(output_dict)
        return -output_dict['quality'] + lambda_tradeoff * max(0, output_dict['cost'] - cost_init)

    def compute_lambdas(self, current_optimal_lambdas, execute_function, cost_target, args_execute_function):
        logger.debug("Computing lambdas with hyperopt strategy")
        self.all_results = []
        space = []
        max_init_val = max([lambda_ for lambda_ in current_optimal_lambdas if lambda_ is not None])
        for i in range(len(current_optimal_lambdas)):
            if current_optimal_lambdas[i] is None and not self.optimize_max_depth:
                space.append(hp.choice(f'lambda_{i}', [None]))
            else:
                max_val = max(0, (1 - self.max_factor) * max_init_val)
                if max_val == 0 or self.from_scratch or self.max_lambda == 1:
                    max_val = self.max_lambda
                space.append(hp.uniform(f'lambda_{i}', 0, max_val))
        
        if self.optimize_max_depth:
            space.append(hp.choice('max_depth', [i for i in range(1, len(current_optimal_lambdas) + 1)]))

        results_init = execute_function(current_optimal_lambdas, *args_execute_function)
        cost_init = results_init['cost']
        lambda_tradeoff = results_init['quality'] / cost_init
        objective = lambda x: self.objective(x, cost_init, lambda_tradeoff, execute_function, *args_execute_function)
        best = fmin(objective, space, algo=tpe.suggest, max_evals=self.n_searches, 
                    rstate= np.random.default_rng(0))
        all_results_low_cost = [result for result in self.all_results if result['cost'] < cost_target]
        if len(all_results_low_cost) == 0:
            return current_optimal_lambdas, cost_init, results_init['quality']
        best_lambda_index = np.argmax([result['quality'] for result in all_results_low_cost])
        best_lambdas = all_results_low_cost[best_lambda_index]['lambdas']
        cost_best = all_results_low_cost[best_lambda_index]['cost']
        quality_best = all_results_low_cost[best_lambda_index]['quality']
        logger.info(f"Final Lambdas: {best_lambdas}")
        logger.info(f"Final Cost: {cost_best}")
        logger.info(f"Final Quality: {quality_best}")
        return best_lambdas, cost_best, quality_best