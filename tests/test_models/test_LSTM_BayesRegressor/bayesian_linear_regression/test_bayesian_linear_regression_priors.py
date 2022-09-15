import random
import pytest
from pydantic import ValidationError

from uncertainty_forecasting.models.regression.LSTM_BayesRegressor.bayesian_linear_regression.bayesian_linear_regression_priors \
    import BayesianLinearRegressionPriors

INTEGER_GENERATION_UPPER_LIMIT = 100


class TestBayesianLinearRegressionParameters():
    """
    self.mean_theta_0 = 1.0
    if priors_thetas is not None:
        self.mean_thetas = 1.0

    self.standard_deviation_theta_0 = 2.0
    self.standard_deviation_thetas = 2.0

    self.standard_deviation_sigma = 10.0
    """

    def test_if_mean_theta_0_not_number_should_raise_type_error(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        invalid_mean_theta_0 = "I'm a string"

        # Action
        with pytest.raises(ValidationError):
            priors.mean_theta_0 = invalid_mean_theta_0

        # Assert

    def test_if_mean_theta_0_is_number_should_set_the_value(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        valid_mean_theta_0 = random.randint(-INTEGER_GENERATION_UPPER_LIMIT,
                                            INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        priors.mean_theta_0 = valid_mean_theta_0

        # Assert
        assert priors.mean_theta_0 == valid_mean_theta_0

    def test_if_mean_thetas_not_number_should_raise_type_error(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        invalid_mean_thetas = "I'm a string"

        # Action
        with pytest.raises(ValidationError):
            priors.mean_thetas = invalid_mean_thetas

        # Assert

    def test_if_mean_thetas_is_number_should_set_the_value(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        valid_mean_thetas = random.randint(-INTEGER_GENERATION_UPPER_LIMIT,
                                           INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        priors.mean_thetas = valid_mean_thetas

        # Assert
        assert priors.mean_thetas == valid_mean_thetas

    def test_if_standard_deviation_theta_0_not_number_should_raise_type_error(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        invalid_standard_deviation_theta_0 = "I'm a string"

        # Action
        with pytest.raises(ValidationError):
            priors.standard_deviation_theta_0 = invalid_standard_deviation_theta_0

        # Assert

    def test_if_standard_deviation_theta_0_is_not_positive_should_raise_value_error(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        invalid_standard_deviation_theta_0 = -1 * random.randint(0, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            priors.standard_deviation_theta_0 = invalid_standard_deviation_theta_0

        # Assert

    def test_if_invalid_standard_deviation_theta_0_is_positive_number_should_set_the_value(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        valid_standard_deviation_theta_0 = random.randint(1, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        priors.standard_deviation_theta_0 = valid_standard_deviation_theta_0

        # Assert
        assert priors.standard_deviation_theta_0 == valid_standard_deviation_theta_0

    def test_if_standard_deviation_thetas_not_number_should_raise_type_error(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        invalid_standard_deviation_thetas = "I'm a string"

        # Action
        with pytest.raises(ValidationError):
            priors.standard_deviation_thetas = invalid_standard_deviation_thetas

        # Assert

    def test_if_standard_deviation_thetas_is_not_positive_should_raise_value_error(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        invalid_standard_deviation_thetas = -1 * random.randint(0, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            priors.standard_deviation_thetas = invalid_standard_deviation_thetas

        # Assert

    def test_if_invalid_standard_deviation_thetas_is_positive_number_should_set_the_value(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        valid_standard_deviation_thetas = random.randint(1, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        priors.standard_deviation_thetas = valid_standard_deviation_thetas

        # Assert
        assert priors.standard_deviation_thetas == valid_standard_deviation_thetas

    def test_if_standard_deviation_sigma_not_number_should_raise_type_error(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        invalid_standard_deviation_sigma = "I'm a string"

        # Action
        with pytest.raises(ValidationError):
            priors.standard_deviation_sigma = invalid_standard_deviation_sigma

        # Assert

    def test_if_standard_deviation_sigma_is_not_positive_should_raise_value_error(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        invalid_standard_deviation_sigma = -1 * random.randint(0, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            priors.standard_deviation_sigma = invalid_standard_deviation_sigma

        # Assert

    def test_if_invalid_standard_deviation_sigma_is_positive_number_should_set_the_value(self):
        # Prepare
        priors = BayesianLinearRegressionPriors()
        valid_standard_deviation_sigma = random.randint(1, INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        priors.standard_deviation_sigma = valid_standard_deviation_sigma

        # Assert
        assert priors.standard_deviation_sigma == valid_standard_deviation_sigma
