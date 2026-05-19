class Algorithm:
    def __init__(self, quality_computer, cost_computer, models, 
                 max_expected_cost, strategies, rounding_digits=8):
        """
        Initializes the BaseAlgorithm object. This serves as the base class for all selection strategies.
        Args:
            quality_computer: The quality computer object used for evaluating model quality.
            cost_computer: The cost computer object used for evaluating model cost.
            models: A list of models to be considered for selection.
            max_expected_cost: The maximum expected cost allowed for the selection strategy.
            strategies: The strategies to be used for hyperparameter optimization.
            rounding_digits: The number of digits to round the results to (default: 8).
        """
        self.quality_computer = quality_computer
        self.cost_computer = cost_computer
        self.models = models
        self.max_expected_cost = max_expected_cost
        self.strategies = strategies
        self.rounding_digits = rounding_digits

    def predict(self, questions, model_answers=None):
        """
        Predicts the model to run for the given questions.

        Args:
            questions (list): A list representing the questions.
            model_answers (list, optional): A list representing the model answers. Defaults to None.

        Returns:
            list: A list of models to run for the given questions.
        """
        raise NotImplementedError
    
    def select_answer(self, questions, model_answers):
        """
        Selects the best answer from a list of model answers based on the given questions.

        Args:
            questions (list): A list representing the questions.
            model_answers (list): A list representing the model answers.

        Raises:
            NotImplementedError: This method is meant to be overridden by subclasses.

        Returns:
            List[str]: The name of the selected model for each question.
        """
        raise NotImplementedError
    
    def fit(self, questions, model_answers, 
            ground_truth_qualities=None, ground_truth_costs=None):
        """
        Fit the algorithm to the given data.
        Args:
            questions: A list of questions.
            model_answers: A list of model answers corresponding to the questions.
            ground_truth_qualities: (optional) A list of ground truth qualities for the model answers.
            ground_truth_costs: (optional) A list of ground truth costs for the model answers.
        """
        raise NotImplementedError