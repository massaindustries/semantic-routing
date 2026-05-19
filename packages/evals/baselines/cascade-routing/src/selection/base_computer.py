class BaseComputer:
    def __init__(self):
        """
        Initializes the BaseComputer object. This is the base class for all computer objects.

        Parameters:
            None

        Returns:
            None
        """
        self.training = False

    def trigger_training(self, training):
        """
        Triggers the training process by setting the training attribute to true or false.
        Args:
            training (boolean): A boolean value indicating whether the training process should be triggered.
        Returns:
            None
        """

        self.training = training

    @property
    def is_independent(self):
        """
        Check if the object returns independent results, i.e., if 
        the resulting values are independent of each other.

        Returns:
            bool: True if the object is independent, False otherwise.
        """
        return True

    def fit(self, questions, model_answers, measure):
        """
        Fits the model to the given questions and model answers using the specified measure.

        Args:
            questions (list): A list of questions.
            model_answers (list): A list of model answers corresponding to the questions.
            measure (list): The value of the measure for each model answer.
        """
        raise NotImplementedError
    
    def predict(self, questions, model_answers=None):
        """
        Make predictions based on the given questions.

        Args:
            questions (list): A list of questions to make predictions on.
            model_answers (list, optional): A list of model answers to make predictions on. Defaults to None.

        Raises:
            NotImplementedError: This method needs to be implemented in a derived class.

        Returns:
            list: A list of predictions. Each question should have a corresponding prediction for each model.
            list: A list of uncertainties. Each question should have a corresponding uncertainty matrix, indicating the variances and covariances of the predictions.
                                            If the uncertainty is None, the prediction is considered to be deterministic.
        """
        raise NotImplementedError