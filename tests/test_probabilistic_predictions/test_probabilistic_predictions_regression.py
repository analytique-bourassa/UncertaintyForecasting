import numpy as np
import pytest

from uncertainty_forecasting.probabilitic_predictions.probabilistic_predictions_regression import ProbabilisticPredictionsRegression

class TestProbabilisticPredictions(object):

    def test_returned_output_labels_with_confidence(self):

        # Prepare
        number_of_predictions = 200
        number_of_samples = 100

        predictions = ProbabilisticPredictionsRegression()
        predictions.number_of_predictions = number_of_predictions
        predictions.number_of_samples = number_of_samples
        predictions.initialize_to_zeros()

        predictions.values = np.random.normal(0, 1, (number_of_predictions, number_of_samples))

        # Action
        y_test_predictions = predictions.predictions

        # Assert
        assert len(y_test_predictions) == number_of_predictions
        assert all([ -1 <= y_predicted <= 1 for y_predicted in y_test_predictions])

    def test_returned_confidence_interval(self):

        # Prepare
        number_of_predictions = 200
        number_of_samples = 100
        standard_deviation = 1
        mean = 0

        predictions = ProbabilisticPredictionsRegression()
        predictions.number_of_predictions = number_of_predictions
        predictions.number_of_samples = number_of_samples
        predictions.initialize_to_zeros()

        predictions.values = np.random.normal(mean, standard_deviation, (number_of_predictions, number_of_samples))

        # Action
        lower, upper = predictions.calculate_confidence_interval(0.95)

        # Assert
        assert len(lower) == number_of_predictions
        assert len(upper) == number_of_predictions

        assert all([ -3*standard_deviation <= lower_value <= 0 for lower_value in lower])
        assert all([ 0 <= upper_value <= 3*standard_deviation for upper_value in upper])

    def test_expect_error_for_interval_bigger_than_one(self):

        # Prepare
        number_of_predictions = 200
        number_of_samples = 100
        standard_deviation = 1
        mean = 0

        predictions = ProbabilisticPredictionsRegression()
        predictions.number_of_predictions = number_of_predictions
        predictions.number_of_samples = number_of_samples
        predictions.initialize_to_zeros()

        predictions.values = np.random.normal(mean, standard_deviation, (number_of_predictions, number_of_samples))

        value_interval_bigger_than_one = 2.0

        # Action
        with pytest.raises(ValueError):
            _, _ = predictions.calculate_confidence_interval(value_interval_bigger_than_one)


