from . import ClassificationCostComputer, GroundTruthQualityComputer, ClassificationQualityComputer, CascadeRouter, Router, ConstantStrategy, HyperoptStrategy, BaselineCascader, GroundTruthCostComputer
from .open_form import OpenFormCostComputer, OpenFormQualityComputer
from .swebench import SWEBenchCostComputer, SWEBenchQualityComputer
from .code_math import CodeMathCostComputer, CodeMathQualityComputer
import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer
from concurrent.futures import as_completed, ProcessPoolExecutor
import os
from collections import Counter
from copy import deepcopy
import time


def convert_to_numpy(model_answers, costs, qualities, models, 
                     is_classification=True, is_routerbench=False):
    """
    Convert the given model answers, costs, and qualities to numpy arrays.

    Args:
        model_answers (pandas.DataFrame): The model answers.
        costs (pandas.DataFrame): The costs.
        qualities (pandas.DataFrame): The qualities.
        models (list): The list of models.
        is_classification (bool, optional): Whether the problem is classification or not. Defaults to True.
        is_routerbench (bool, optional): Whether the data is from RouterBench or not. Defaults to False.

    Returns:
        tuple: A tuple containing the converted model answers, costs, and qualities as numpy arrays.
    """
    # if they are already numpy arrays, return them
    model_answers = model_answers[models].values
    if not is_routerbench:
        for i in range(len(model_answers)):
            for j in range(len(model_answers[i])):
                if is_classification:
                    model_answers[i, j] = np.array(model_answers[i, j])
                else:
                    model_answers[i, j][1] = np.array(model_answers[i, j][1])
    costs = costs[models].values
    qualities = qualities[models].values
    return model_answers, costs, qualities

def prediction(cascader, questions, qualities, costs, 
               actual_answers, models, is_router=False):
    """
    Perform model selection based on cascader predictions.

    Args:
        cascader (object): The cascader/router/cascade router object used for predictions.
        questions (list): List of questions to be predicted.
        qualities (list): List of qualities for each question and model.
        costs (list): List of costs for each question and model.
        actual_answers (list): List of actual answers for each question and model.
        models (list): List of available models.
        is_router (bool, optional): Flag indicating if cascader is used as a router. Defaults to False.

    Returns:
        dict: A dictionary containing the following information:
            - 'quality': The mean quality of the selected models.
            - 'cost': The mean cost of the selected models.
            - 'models_run': A counter object with the count of models run for each question.
            - 'selected_models': A counter object with the count of times each model was selected.
            - 'lambdas': A list of lambdas used by the cascader.
            - 'mean_times': The mean time taken for predictions.
            - 'median_times': The median time taken for predictions.
            - 'max_times': The maximum time taken for predictions.
    """
    qualities_output = []
    costs_output = []
    models_run = []
    selected_models = []
    timings = []
    for i, question in enumerate(questions):
        model_answers = [[None for _ in range(len(qualities[i]))]]
        cost = 0
        models_run_question = []
        for step in range(len(model_answers[0])):
            t = time.time()
            model = cascader.predict([question], model_answers)
            timings.append(time.time() - t)
            if model[0] is None:
                break
            else:
                model_index = models.index(model[0])
                models_run_question.append(model_index)
                model_answers[0][model_index] = actual_answers[i][model_index]
                cost += costs[i][model_index]
            if is_router:
                break
        selected_answer = cascader.select_answer([question], model_answers)
        selected_model = models.index(selected_answer[0])
        quality = qualities[i][selected_model]
        qualities_output.append(quality)
        costs_output.append(cost)
        models_run.append(','.join([str(model) for model in models_run_question]))
        selected_models.append(selected_model)
    counter = Counter(models_run)
    counter_selected = Counter(selected_models)
    return {
        'quality': np.mean(qualities_output),
        'cost': np.mean(costs_output),
        'models_run': counter,
        'selected_models': counter_selected,
        'lambdas': list(cascader.get_lambdas()),
        'mean_times': float(np.mean(timings)),
        'median_times': float(np.median(timings)),
        'max_times': float(np.max(timings)),
        "quality_all": qualities_output,
        "cost_all": costs_output,
    }

def remove_redundant_models(qualities, costs, model_indices=None):
    """
    Remove models that are not on the pareto-frontier based on their qualities and costs.

    Args:
        qualities (list): List of qualities for each model.
        costs (list): List of costs for each model.
        model_indices (list, optional): List of indices for each model. Defaults to None.

    Returns:
        tuple: A tuple containing the updated qualities, costs, and model indices after removing redundant models.
    """
    if model_indices is None:
        model_indices = list(range(len(qualities)))
    for i in range(len(qualities)):
        for j in range(len(qualities)):
            for k in range(len(qualities)):
                if costs[i] <= costs[k] <= costs[j] and i != j and i != k and j != k:
                    quality_at_k = (qualities[i] - qualities[j]) * (costs[k] - costs[j]) / (costs[i] - costs[j]) + qualities[j]
                    if qualities[k] < quality_at_k:
                        return remove_redundant_models(
                            [qualities[l] for l in range(len(qualities)) if l != k],
                            [costs[l] for l in range(len(qualities)) if l != k],
                            [model_indices[l] for l in range(len(qualities)) if l != k]
                        )
                if costs[j] < costs[k] and j != k and qualities[j] > qualities[k]:
                    return remove_redundant_models(
                        [qualities[l] for l in range(len(qualities)) if l != k],
                        [costs[l] for l in range(len(qualities)) if l != k],
                        [model_indices[l] for l in range(len(qualities)) if l != k]
                    )
    cost_indices = np.argsort(costs)
    qualities = [qualities[i] for i in cost_indices]
    costs = [costs[i] for i in cost_indices]
    model_indices = [model_indices[i] for i in cost_indices]
    return qualities, costs, model_indices

def area_under_curve(y, x, x_min, x_max, y_min_base):
    """
    Calculates the area under the curve defined by the given x and y values.
    Parameters:
    - y (array-like): The y values of the curve.
    - x (array-like): The x values of the curve.
    - x_min (float): The minimum x value for calculating the area.
    - x_max (float): The maximum x value for calculating the area.
    - y_min_base (float): The base y value for extrapolation.
    Returns:
    - float: The area under the curve between x_min and x_max.
    """
    x = np.array(x)
    y = np.array(y)
    # drop indices where (x_i, y_i) is a duplicate
    combined = np.array(list(zip(x, y)))
    unique = np.unique(combined, axis=0)
    x = unique[:, 0]
    y = unique[:, 1]
    # sort by x
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")

    if x[-1] < x_max:
        x = np.append(x, x_max)
        y = np.append(y, y[-1])
    if x[0] > x_min:
        x = np.append(x, x_min)
        y = np.append(y, y_min_base)
    
    if x_min not in x:
        x = np.append(x, x_min)
        y = np.append(y, interp_func(x_min))
    
    if x_max not in x:
        x = np.append(x, x_max)
        y = np.append(y, interp_func(x_max))
    
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    
    mask = (x >= x_min) & (x <= x_max)
    x_filtered = x[mask]
    y_filtered = y[mask]
    area = np.trapz(y_filtered, x_filtered)
    
    return area / (x_max - x_min)

def max_diff(qualities, costs, baseline_qualities, baseline_costs):
    """
    Calculate the maximum and minimum difference between the qualities of 
    models and their interpolated qualities based on baseline models.

    Parameters:
    - qualities (list): List of qualities of the models.
    - costs (list): List of costs associated with the models.
    - baseline_qualities (list): List of qualities of the baseline models.
    - baseline_costs (list): List of costs associated with the baseline models.

    Returns:
    - max_diff (float): Maximum difference between the qualities of the models and their interpolated qualities.
    - min_diff (float): Minimum difference between the qualities of the models and their interpolated qualities.
    """
    baseline_qualities, baseline_costs, _ = remove_redundant_models(baseline_qualities, baseline_costs)
    max_diff = 0
    min_diff = 1000
    interp_func = interp1d(baseline_costs, baseline_qualities, kind='linear', fill_value="extrapolate")
    for i in range(len(qualities)):
        interpol_quality = interp_func(costs[i])
        diff = qualities[i] - interpol_quality
        if diff > max_diff:
            max_diff = diff
        if diff < min_diff:
            min_diff = diff
    return max_diff, min_diff

def auc_all(qualities, costs, baseline_qualities, baseline_costs):
    """
    Calculate the area under the curve (AUC) and the maximum difference between qualities and costs.

    Args:
        qualities (list): List of qualities.
        costs (list): List of costs.
        baseline_qualities (list): List of baseline qualities.
        baseline_costs (list): List of baseline costs.

    Returns:
        dict: A dictionary containing the AUC and the maximum difference.

    """
    cheapest = min(baseline_costs)
    most_expensive = max(baseline_costs)
    cheapest_quality = min(baseline_qualities)
    auc = area_under_curve(qualities, costs, cheapest, most_expensive, cheapest_quality)
    return {
        'auc': auc,
        'max_diff': max_diff(qualities, costs, baseline_qualities, baseline_costs),
    }

def test_router(models, max_cost, train_model_answers, train_costs, train_qualities, train_queries,
                test_model_answers, test_costs, test_qualities, test_queries, data_folder, dataset,
                assume_constant=False, model_class=LogisticRegression, 
                n_highest_include=2, train_split=800, max_lambda=10000, 
                is_router=False, greedy=False, is_cascader=False, 
                is_cascader_ours=False, force_order=False,
                max_depth=None, n_samples=100, 
                ground_truth_noise_before=None, ground_truth_noise_after=None, 
                do_speedup=True, is_latency_cost=False,
                set_sigma_none=False, is_classification=True, 
                cost_noise_before=None, cost_noise_after=None,
                is_swebench=False,
                is_code_math=False,
                access_tests=True,
                ground_truth_cost_computer=False, is_routerbench=False,
                cascade_strategies=[
                    lambda max_lambda: ConstantStrategy(max_lambda, n_iterations=30),
                    lambda max_lambda: HyperoptStrategy(max_lambda, 100, max_factor=4),
                    lambda max_lambda: HyperoptStrategy(max_lambda, 100, max_factor=4)
                ], cascade_router_strategies=[
                    lambda max_lambda: ConstantStrategy(max_lambda, n_iterations=30),
                    lambda max_lambda: HyperoptStrategy(max_lambda, 100, max_factor=4),
                    lambda max_lambda: HyperoptStrategy(max_lambda, 100, max_factor=4)
                ]):
    """
    Performs model selection using the specified parameters.
    Args:
        models (list): List of models to consider for selection.
        max_cost (float): Maximum cost allowed for the selected model.
        train_model_answers (list): List of model answers for training data.
        train_costs (list): List of costs for training data.
        train_qualities (list): List of qualities for training data.
        train_queries (list): List of queries for training data.
        test_model_answers (list): List of model answers for testing data.
        test_costs (list): List of costs for testing data.
        test_qualities (list): List of qualities for testing data.
        test_queries (list): List of queries for testing data.
        data_folder (str): Path to the data folder.
        dataset (str): Name of the dataset.
        assume_constant (bool, optional): Whether to assume constant cost. Defaults to False.
        model_class (class, optional): Model class to use. Defaults to LogisticRegression.
        n_highest_include (int, optional): Number of highest quality models to include for classification. Defaults to 2.
        train_split (int, optional): Index to split the training data for training the linear model and optimizing the hyperparameters. Defaults to 800.
        max_lambda (int, optional): Maximum lambda value. Defaults to 10000.
        is_router (bool, optional): Whether to use router strategy. Defaults to False.
        greedy (bool, optional): Whether to use greedy strategy. Defaults to False.
        is_cascader (bool, optional): Whether to use cascader strategy. Defaults to False.
        is_cascader_ours (bool, optional): Whether to use our cascader strategy. Defaults to False.
        force_order (bool, optional): Whether to force order of models in cascade routing. Defaults to False.
        max_depth (int, optional): Maximum depth for cascade routing strategy. Defaults to None.
        n_samples (int, optional): Number of samples for quality computation. Defaults to 100.
        ground_truth_noise_before (float, optional): Noise before run for ground truth quality computation. Defaults to None.
        ground_truth_noise_after (float, optional): Noise after run for ground truth quality computation. Defaults to None.
        do_speedup (bool, optional): Whether to use speedup in cascade router. Defaults to True.
        set_sigma_none (bool, optional): Whether to set quality deviations to none to None. Defaults to False.
        is_classification (bool, optional): Whether to problem is a classification problem. Defaults to True.
        cost_noise_before (float, optional): Noise before run for cost computation. Defaults to None.
        cost_noise_after (float, optional): Noise after run for cost computation. Defaults to None.
        ground_truth_cost_computer (bool, optional): Whether to use ground truth cost computation. Defaults to False.
        is_routerbench (bool, optional): Whether the problem is the routerbench problem. Defaults to False.
        cascade_strategies (list, optional): List of hyperparameter optimization strategies for cascader. Defaults to [ConstantStrategy, HyperoptStrategy, HyperoptStrategy].
        cascade_router_strategies (list, optional): List of hyperparameter optimization strategies for cascade router. Defaults to [ConstantStrategy, HyperoptStrategy, HyperoptStrategy].
    Returns:
        tuple: A tuple containing the test results and train results.
    """
    model_names = [model['name'] for model in models]
    train_model_answers_here, train_costs_here, train_qualities_here = convert_to_numpy(train_model_answers, 
                                                                                        train_costs, 
                                                                                        train_qualities, 
                                                                                        model_names, 
                                                                                        is_classification, 
                                                                                        is_routerbench)
    test_model_answers_here, test_costs_here, test_qualities_here = convert_to_numpy(test_model_answers, 
                                                                                     test_costs, 
                                                                                     test_qualities, 
                                                                                     model_names, 
                                                                                     is_classification, 
                                                                                     is_routerbench)
    

    if ground_truth_cost_computer:
        cost_computer = GroundTruthCostComputer(
            cost_noise_before, cost_noise_after, assume_constant=assume_constant
        )
    elif is_classification:
        model_names_huggingface = [model.get('huggingface_name', model['name']) for model in models]
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in model_names_huggingface]
        cost_computer = ClassificationCostComputer(
            input_costs=[model['read_cost'] for model in models],
            output_costs=[model['write_cost'] for model in models],
            tokenizers=tokenizers,
            constant_cost=assume_constant,
            is_latency_cost=is_latency_cost,
            store_all=True
        )
    elif is_swebench:
        cost_computer = SWEBenchCostComputer(
            store_all=True,
            constant_cost=assume_constant,
        )
    elif is_code_math:
        cost_computer = CodeMathCostComputer(
            store_all=True,
            constant_cost=assume_constant,
        )
    else:
        model_names_huggingface = [model.get('huggingface_name', model['name']) for model in models]
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in model_names_huggingface]
        cost_computer = OpenFormCostComputer(
            input_costs=[model['read_cost'] for model in models],
            output_costs=[model['write_cost'] for model in models],
            tokenizers=tokenizers,
            constant_cost=assume_constant,
            is_latency_cost=is_latency_cost,
            store_all=True
        )

    if ground_truth_noise_before is None:
        if is_swebench:
            quality_computer = SWEBenchQualityComputer(
                max_depth=(max_depth if not is_cascader else None),
                n_samples=n_samples,
                require_constant_not_run=is_cascader,
                access_tests=access_tests
            )
        elif is_code_math:
            quality_computer = CodeMathQualityComputer(
                max_depth=(max_depth if not is_cascader else None),
                n_samples=n_samples,
                require_constant_not_run=is_cascader
            )
        else:
            quality_class = ClassificationQualityComputer if is_classification else OpenFormQualityComputer
            quality_computer = quality_class(
                model_class=model_class,
                n_highest_include=n_highest_include,
                require_constant_not_run=is_cascader,
                add_entropy=True,
                add_equal_argmax=True,
                add_js_divergence=True,
                n_samples=n_samples,
                store_all=True,
                max_depth=(max_depth if not is_cascader else None),
            )
    else:
        quality_computer = GroundTruthQualityComputer(
            noise_before_run=ground_truth_noise_before,
            noise_after_run=ground_truth_noise_after,
        )

    if not ground_truth_cost_computer:
        cost_computer.fit(train_queries[:train_split], train_model_answers_here[:train_split], 
                          train_costs_here[:train_split])
    else:
        cost_computer.fit(
            np.concatenate([train_queries, test_queries], axis=0),
            np.concatenate([train_model_answers_here, test_model_answers_here], axis=0),
            np.concatenate([train_costs_here, test_costs_here], axis=0)
        )
    if is_latency_cost:
        cost_computer.add_latency_ground_truths(
            np.concatenate([train_queries, test_queries], axis=0),
            np.concatenate([train_model_answers_here, test_model_answers_here], axis=0),
            np.concatenate([train_costs_here, test_costs_here], axis=0)
        )
    if ground_truth_noise_before is None:
        quality_computer.fit(train_queries[:train_split], train_model_answers_here[:train_split], 
                             train_qualities_here[:train_split])
    else:
        quality_computer.fit(
            np.concatenate([train_queries, test_queries], axis=0),
            np.concatenate([train_model_answers_here, test_model_answers_here], axis=0),
            np.concatenate([train_qualities_here, test_qualities_here], axis=0)
        )
    if is_cascader:
        max_lambda = 1

    if not is_router and not is_cascader:
        strategies = [strategy(max_lambda) for strategy in cascade_router_strategies]
    elif is_cascader:
        strategies = [strategy(max_lambda) for strategy in cascade_strategies]
    else:
        strategies=[
            ConstantStrategy(max_lambda, n_iterations=100)
        ]

    if is_cascader:
        router = BaselineCascader(
            cost_computer=cost_computer,
            quality_computer=quality_computer,
            models=model_names,
            max_expected_cost=max_cost,
            strategies=strategies,
        )
    elif is_router:
        router = Router(
            cost_computer=cost_computer,
            quality_computer=quality_computer,
            models=model_names,
            max_expected_cost=max_cost,
            strategies=strategies,
            rounding_digits=6,
        )
    else:
        router = CascadeRouter(
            cost_computer=cost_computer,
            quality_computer=quality_computer,
            models=model_names,
            max_expected_cost=max_cost,
            strategies=strategies,
            rounding_digits=6,
            greedy=greedy,
            force_order=force_order,
            max_depth=(max_depth if not is_cascader_ours else None),
            set_sigma_none=set_sigma_none,
            cascade=is_cascader_ours,
            do_speedup=do_speedup
        )

    if is_routerbench:
        router.fit(train_queries, train_model_answers_here, train_qualities_here, train_costs_here)
    else:
        router.fit(train_queries[train_split:], train_model_answers_here[train_split:], 
               train_qualities_here[train_split:], train_costs_here[train_split:])

    test_results = prediction(router, test_queries, test_qualities_here, 
                              test_costs_here, test_model_answers_here, model_names, 
                              is_router=is_router)
    train_results = prediction(router, train_queries[train_split:], train_qualities_here[train_split:], 
                               train_costs_here[train_split:], 
                               train_model_answers_here[train_split:], model_names, is_router=is_router)

    return test_results, train_results

def test_router_all(
        models,
        max_costs,
        n_cores=8,
        **kwargs
):  
    """
    Perform testing and training on multiple models using a range of maximum costs.
    Args:
        models (list): List of models to be tested.
        max_costs (list): List of maximum costs for testing.
        n_cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to 8.
        **kwargs: Additional keyword arguments to be passed to the test_router function.
    Returns:
        tuple: A tuple containing the test results and train results for all models and maximum costs.
    """
    test_results = []
    train_results = []

    if n_cores == 1:
        for max_cost in max_costs:
            test_result, train_result = test_router(models, max_cost, **kwargs)
            test_results.append(test_result)
            train_results.append(train_result)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            future_to_cost = {executor.submit(test_router, models, max_cost, **kwargs): max_cost for max_cost in max_costs}
            
            for future in as_completed(future_to_cost):
                test_result, train_result = future.result()
                test_results.append(test_result)
                train_results.append(train_result)
    
    all_results_test = prepare_results(test_results)

    all_results_train = prepare_results(train_results)
    
    return all_results_test, all_results_train

def prepare_results(results):
    """
    Prepare the results for model selection.

    Args:
        results (list): A list of dictionaries containing the results for each model.

    Returns:
        dict: A dictionary containing the prepared results with sorted cost values.

    """

    all_results_test = {
        'quality': [result['quality'] for result in results],
        'cost': [result['cost'] for result in results],
        'models_run': [result['models_run'] for result in results],
        'selected_models': [result['selected_models'] for result in results],
        'lambdas': [result['lambdas'] for result in results],
        'mean_times': [result['mean_times'] for result in results],
        'median_times': [result['median_times'] for result in results],
        'max_times': [result['max_times'] for result in results],
        "quality_all": [result["quality_all"] for result in results],
        "cost_all": [result["cost_all"] for result in results],
    }

    all_results_indices = np.argsort(all_results_test['cost'])
    all_results_test['cost'] = np.array(all_results_test['cost'])[all_results_indices].tolist()
    all_results_test['quality'] = np.array(all_results_test['quality'])[all_results_indices].tolist()
    all_results_test['models_run'] = np.array(all_results_test['models_run'])[all_results_indices].tolist()
    all_results_test['selected_models'] = np.array(all_results_test['selected_models'])[all_results_indices].tolist()
    all_results_test['lambdas'] = np.array(all_results_test['lambdas'])[all_results_indices].tolist()
    all_results_test['mean_times'] = np.array(all_results_test['mean_times'])[all_results_indices].tolist()
    all_results_test['median_times'] = np.array(all_results_test['median_times'])[all_results_indices].tolist()
    all_results_test['max_times'] = np.array(all_results_test['max_times'])[all_results_indices].tolist()
    all_results_test['quality_all'] = np.array(all_results_test['quality_all'])[all_results_indices].tolist()
    all_results_test['cost_all'] = np.array(all_results_test['cost_all'])[all_results_indices].tolist()
    return all_results_test
                
def test_everything(models, test_costs_averaged, test_qualities_averaged, n_iterations=5, 
                     no_router=False, no_cascade=False, no_cascade_router=False,
                     assume_constant=False, use_sum_costs=False,
                     **kwargs):
    """
    Test the performance of different models using various configurations.
    Args:
        models (list): List of models to use for model selected.
        test_costs_averaged (dict): Dictionary mapping model names to their averaged test costs.
        test_qualities_averaged (dict): Dictionary mapping model names to their averaged test qualities.
        n_iterations (int, optional): Number of iterations for linear spacing. Defaults to 5.
        no_router (bool, optional): Flag to exclude router testing. Defaults to False.
        no_cascade (bool, optional): Flag to exclude cascade testing. Defaults to False.
        no_cascade_router (bool, optional): Flag to exclude cascade router testing. Defaults to False.
        **kwargs: Additional keyword arguments.
    Returns:
        dict: Dictionary containing the test results and metrics.
            - 'test': Test results for all models.
            - 'train': Training results for all models.
            - 'cascade_test': Cascade test results.
            - 'cascade_train': Cascade training results.
            - 'cascade_test_ours': Our cascade test results.
            - 'cascade_train_ours': Our cascade training results.
            - 'router_test': Router test results.
            - 'router_train': Router training results.
            - 'qualities_baseline': List of baseline qualities.
            - 'costs_baseline': List of baseline costs.
            - 'aucs': AUC scores.
            - 'aucs_router': AUC scores for router testing.
            - 'aucs_baseline': AUC scores for baseline testing.
            - 'aucs_cascade': AUC scores for cascade testing.
            - 'aucs_cascade_ours': AUC scores for our cascade testing.
    """
    model_names = [model['name'] for model in models]
    if not use_sum_costs:
        sorted_costs = sorted([test_costs_averaged[name] for name in model_names])
        _, costs_not_redundant, _ = remove_redundant_models([test_qualities_averaged[name] for name in model_names], 
                                                            [test_costs_averaged[name] for name in model_names])
        sorted_costs = sorted(costs_not_redundant)
        lin_spaces = [
            np.linspace(sorted_costs[i], sorted_costs[i+1], n_iterations) for i in range(len(sorted_costs) - 1)
        ]
        max_cost_space = np.concatenate(lin_spaces)
    else:
        costs_models = [test_costs_averaged[name] for name in model_names]
        max_cost_space = np.linspace(min(costs_models), sum(costs_models), n_iterations * len(models))
    
    if no_cascade_router:
        results, results_train = None, None
    else:
        results, results_train = test_router_all(models, max_cost_space, assume_constant=False or assume_constant, 
                                                 is_cascader=False, is_router=False, **kwargs)
    
    if no_cascade:
        results_cascade, results_cascade_train = None, None
        results_cascade_ours, results_cascade_train_ours = None, None
    else:
        results_cascade, results_cascade_train = test_router_all(models, max_cost_space, assume_constant=True, 
                                                                is_cascader=True, **kwargs)
        if not no_cascade_router:
            results_cascade_ours, results_cascade_train_ours = test_router_all(models, max_cost_space, 
                                                                           assume_constant=False or assume_constant,
                                                                            is_cascader_ours=True, **kwargs)
        else:
            results_cascade_ours, results_cascade_train_ours = None, None
    
    if no_router:
        results_router, results_router_train = None, None
    else:
        results_router, results_router_train = test_router_all(models, max_cost_space, 
                                                               assume_constant=False or assume_constant, 
                                                               is_router=True, **kwargs)

    qualities_baseline = np.array(np.array(test_qualities_averaged[model_names]))
    costs_baseline = np.array(np.array(test_costs_averaged[model_names]))
    if no_cascade_router:
        aucs = None
    else:
        aucs = auc_all(results['quality'], results['cost'], qualities_baseline, costs_baseline)

    if no_cascade:
        aucs_cascade = None
        aucs_cascade_ours = None
    else:
        aucs_cascade = auc_all(results_cascade['quality'], results_cascade['cost'], qualities_baseline, costs_baseline)
        if no_cascade_router:
            aucs_cascade_ours = None
        else:
            aucs_cascade_ours = auc_all(results_cascade_ours['quality'], results_cascade_ours['cost'], qualities_baseline, costs_baseline)

    if no_router:
        aucs_router = None
    else:
        aucs_router = auc_all(results_router['quality'], results_router['cost'], qualities_baseline, costs_baseline)

    qualities_baseline_removed, costs_baseline_removed, _ = remove_redundant_models(
        [test_qualities_averaged[model['name']] for model in models], 
        [test_costs_averaged[model['name']] for model in models]
    )
    aucs_baseline = auc_all(qualities_baseline_removed, costs_baseline_removed, 
                            qualities_baseline, costs_baseline)
    return {
        'test': results,
        'train': results_train,
        'cascade_test': results_cascade,
        'cascade_train': results_cascade_train,
        'cascade_test_ours': results_cascade_ours,
        'cascade_train_ours': results_cascade_train_ours,
        'router_test': results_router,
        'router_train': results_router_train,
        'qualities_baseline': qualities_baseline.tolist(),
        'costs_baseline': costs_baseline.tolist(),
        'aucs': aucs,
        'aucs_router': aucs_router,
        'aucs_baseline': aucs_baseline,
        'aucs_cascade': aucs_cascade,
        'aucs_cascade_ours': aucs_cascade_ours
    }