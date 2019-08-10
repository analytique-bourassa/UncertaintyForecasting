import numpy as np

from probabilitic_predictions.probabilistic_predictions_classification import ProbabilisticPredictionsClassification

class TestProbabilisticPredictions(object):

    def test_returned_output_labels_with_confidence(self):

        # Prepare
        number_of_predictions = 100
        number_of_samples =20
        number_of_classes = 7

        predictions = ProbabilisticPredictionsClassification(number_of_classes=number_of_classes)
        predictions.number_of_predictions = number_of_predictions
        predictions.number_of_samples = number_of_samples
        predictions.initialize_to_zeros()

        predictions.values = np.random.randint(0, number_of_classes, (number_of_predictions, number_of_samples))

        # Action
        y_test_predictions, confidences = predictions.predictions_with_confidence

        # Assert
        assert len(y_test_predictions) == number_of_predictions
        assert len(confidences) == number_of_predictions


