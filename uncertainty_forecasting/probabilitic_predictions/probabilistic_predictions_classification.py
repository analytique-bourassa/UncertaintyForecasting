import numpy as np
import matplotlib.pyplot as plt

from uncertainty_forecasting.utils.validator import Validator
DEFAULT_LEGEND_SIZE = 19

class ProbabilisticPredictionsClassification():

    KEY_INT_FOR_SAMPLES_PREDICTIONS = 1
    KEY_INT_FOR_PREDICTIONS_PREDICTIONS = 0
    KEY_INT_TRUE_VALUES = 0
    KEY_FOR_NUMBER_OF_TRAINING_DATA = 0
    KEY_INDEX_FOR_NUMBER_OF_DATA = 0

    def __init__(self, number_of_classes, classes_names=None):

        self._number_of_classes = None
        self._number_of_samples = None
        self._number_of_predictions = None
        self._values = None
        self._true_values = None

        self._train_data = None

        self.number_of_classes = number_of_classes
        self._classes_names = classes_names


    def initialize_to_zeros(self):

        assert self.number_of_predictions is not None
        assert self.number_of_samples is not None

        self.values = np.zeros((self.number_of_predictions, self.number_of_samples))
        self.true_values = np.zeros(self.number_of_predictions)

    @property
    def number_of_classes(self):
        return self._number_of_classes

    @number_of_classes.setter
    def number_of_classes(self, value):
        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._number_of_classes = value

    @property
    def number_of_samples(self):
        return self._number_of_samples

    @number_of_samples.setter
    def number_of_samples(self, value):
        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._number_of_samples = value

    @property
    def number_of_predictions(self):
        return self._number_of_predictions

    @number_of_predictions.setter
    def number_of_predictions(self, value):
        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._number_of_predictions = value

    @property
    def values(self):
        return self._values.astype(np.int64)

    @values.setter
    def values(self, value):

        assert self.number_of_samples is not None, "The number of samples must have been setted."
        assert self.number_of_predictions is not None, "The number of predictions must have been setted."

        assert isinstance(value, np.ndarray)
        assert value.shape[self.KEY_INT_FOR_SAMPLES_PREDICTIONS] == self.number_of_samples
        assert value.shape[self.KEY_INT_FOR_PREDICTIONS_PREDICTIONS] == self.number_of_predictions

        self._values = value

    @property
    def true_values(self):
        return self._true_values

    @true_values.setter
    def true_values(self, value):
        assert self.number_of_predictions is not None, "The number of predictions must have been setted."

        assert isinstance(value, np.ndarray)
        assert value.shape[self.KEY_INT_TRUE_VALUES] == self.number_of_predictions

        self._true_values = value

    @property
    def predictions(self):

        number_of_data_to_predict = self.values.shape[self.KEY_INDEX_FOR_NUMBER_OF_DATA]

        predictions = np.zeros(number_of_data_to_predict, dtype=np.int64)

        for index in range(number_of_data_to_predict):

            counts = np.bincount(self.values[index])
            label = np.argmax(counts)
            predictions[index] = label

        return predictions

    @property
    def predictions_with_confidence(self):

        number_of_data_to_predict = self.values.shape[self.KEY_INDEX_FOR_NUMBER_OF_DATA]

        predictions = np.zeros(number_of_data_to_predict, dtype=np.int64)
        confidences = np.zeros((number_of_data_to_predict, self.number_of_classes))

        for index in range(number_of_data_to_predict):

            counts_per_class = np.bincount(self.values[index])

            label = np.argmax(counts_per_class)
            confidence_per_class = counts_per_class/counts_per_class.sum()

            predictions[index] = label
            confidences[index, :len(confidence_per_class)] = confidence_per_class

        return predictions, confidences

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, value):
        assert isinstance(value, np.ndarray)

        self._train_data = value



