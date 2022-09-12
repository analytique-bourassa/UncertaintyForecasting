import random
import pytest

from uncertainty_forecasting.models.regression.LSTM_BayesRegressor.bayesian_linear_regression.bayesian_linear_regression_parameters \
    import BayesianLinearRegressionParameters

INTEGER_GENERATION_UPPER_LIMIT = 100


class TestBayesianLinearRegressionParameters():

    """
    self.number_of_samples_for_predictions = 1000 if not SMOKE_TEST else 1
    self.number_of_samples_for_posterior = 10000 if not SMOKE_TEST else 1
    self.number_of_tuning_steps = 1000 if not SMOKE_TEST else 1
    self.number_of_iterations = 500000 if not SMOKE_TEST else 1
    """


    def test_if_number_sample_for_predictions_not_positive_should_raise_value_error(self):

        # Prepare
        params = BayesianLinearRegressionParameters()
        invalid_number_of_samples_for_predictions = -1*random.randint(0,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            params.number_of_samples_for_predictions = invalid_number_of_samples_for_predictions

        # Assert

    def test_if_number_sample_for_predictions_positive_integer_should_set_the_value(self):
        # Prepare
        params = BayesianLinearRegressionParameters()
        valid_number_of_samples_for_predictions = random.randint(1,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        params.number_of_samples_for_predictions = valid_number_of_samples_for_predictions

        # Assert
        assert params.number_of_samples_for_predictions == valid_number_of_samples_for_predictions

    def test_if_number_of_samples_for_posterior_not_positive_should_raise_value_error(self):

        # Prepare
        params = BayesianLinearRegressionParameters()
        invalid_number_of_samples_for_posterior = -1*random.randint(0,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            params.number_of_samples_for_posterior = invalid_number_of_samples_for_posterior

        # Assert

    def test_if_number_of_samples_for_posterior_positive_integer_should_set_the_value(self):
        # Prepare
        params = BayesianLinearRegressionParameters()
        valid_number_of_samples_for_posterior = random.randint(1,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        params.number_of_samples_for_posterior = valid_number_of_samples_for_posterior

        # Assert
        assert params.number_of_samples_for_posterior == valid_number_of_samples_for_posterior

    def test_if_number_of_tuning_steps_not_positive_should_raise_value_error(self):
        # Prepare
        params = BayesianLinearRegressionParameters()
        invalid_number_of_tuning_steps = -1 * random.randint(0, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            params.number_of_tuning_steps = invalid_number_of_tuning_steps

        # Assert

    def test_if_number_of_tuning_steps_positive_integer_should_set_the_value(self):
        # Prepare
        params = BayesianLinearRegressionParameters()
        valid_number_of_tuning_steps = random.randint(1, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        params.number_of_tuning_steps = valid_number_of_tuning_steps

        # Assert
        assert params.number_of_tuning_steps == valid_number_of_tuning_steps

    def test_if_number_of_iterations_not_positive_should_raise_value_error(self):
        # Prepare
        params = BayesianLinearRegressionParameters()
        invalid_number_of_iterations = -1 * random.randint(0, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            params.number_of_iterations = invalid_number_of_iterations

        # Assert

    def test_if_number_of_iterations_positive_integer_should_set_the_value(self):
        # Prepare
        params = BayesianLinearRegressionParameters()
        valid_number_of_iterations = random.randint(1, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        params.number_of_iterations = valid_number_of_iterations

        # Assert
        assert params.number_of_iterations == valid_number_of_iterations
