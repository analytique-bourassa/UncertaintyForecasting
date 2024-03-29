import numpy as np

from uncertainty_forecasting.probabilitic_predictions.probabilistic_predictions_classification import ProbabilisticPredictionsClassification
from uncertainty_forecasting.models.calibration.diagnostics import calculate_static_calibration_error
class TestDiagnostics():

    def test_if_shape_of_predictions_is_good_should_return_calibration_curve(self):

        # Prepare
        number_of_classes = 6
        number_of_samples_for_predictions = 100
        number_of_predictions = 1000

        predictions = ProbabilisticPredictionsClassification(number_of_classes=number_of_classes)
        predictions.number_of_predictions = number_of_predictions
        predictions.number_of_samples = number_of_samples_for_predictions
        predictions.initialize_to_zeros()

        predictions.values = np.random.randint(0, number_of_classes, (number_of_predictions
                                                                    ,number_of_samples_for_predictions))
        predictions.true_values = np.random.randint(0, number_of_classes, number_of_predictions)
        y_test_predictions, confidences = predictions.predictions_with_confidence

        # action

        curve, means_per_bin, deviation_score = calculate_static_calibration_error(y_test_predictions,
                                                                             predictions.true_values,
                                                                             confidences,
                                                                             predictions.number_of_classes)
        # assert
        assert  isinstance(curve, np.ndarray)
        assert isinstance(means_per_bin, np.ndarray)
        assert deviation_score >= 0