from uncertainty_forecasting.utils.validator import Validator

class BayesianLinearRegressionParameters():

    def __init__(self, SMOKE_TEST=False):

        self._number_of_samples_for_predictions = None
        self._number_of_samples_for_posterior = None
        self._number_of_tuning_steps = None
        self._number_of_iterations = None

        self.number_of_samples_for_predictions = 1000 if not SMOKE_TEST else 1
        self.number_of_samples_for_posterior = 10000 if not SMOKE_TEST else 1
        self.number_of_tuning_steps = 1000 if not SMOKE_TEST else 1
        self.number_of_iterations = 500000 if not SMOKE_TEST else 1

    @property
    def number_of_samples_for_predictions(self):
        return self._number_of_samples_for_predictions

    @number_of_samples_for_predictions.setter
    def number_of_samples_for_predictions(self, value):

        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._number_of_samples_for_predictions = value

    @property
    def number_of_samples_for_posterior(self):
        return self._number_of_samples_for_posterior

    @number_of_samples_for_posterior.setter
    def number_of_samples_for_posterior(self, value):

        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._number_of_samples_for_posterior = value

    @property
    def number_of_tuning_steps(self):
        return self._number_of_tuning_steps

    @number_of_tuning_steps.setter
    def number_of_tuning_steps(self, value):

        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._number_of_tuning_steps = value

    @property
    def number_of_iterations(self):
        return self._number_of_iterations

    @number_of_iterations.setter
    def number_of_iterations(self, value):

        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._number_of_iterations = value








