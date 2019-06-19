import numpy as np

class ProbabilisticPredicitons():

    KEY_INT_FOR_SAMPLES_PREDICTIONS = 0
    KEY_INT_FOR_PREDICTIONS_PREDICTIONS = 1
    KEY_INT_TRUE_VALUES = 0

    def __init__(self):

        self._number_of_samples = None
        self._number_of_predictions = None
        self._values = None
        self._true_values = None

    @property
    def number_of_samples(self):
        return self._number_of_samples

    @number_of_samples.setter
    def number_of_samples(self, value):
        assert isinstance(value, int)
        assert value > 0

        self._number_of_samples = value

    @property
    def number_of_predictions(self):
        return self._number_of_predictions

    @number_of_predictions.setter
    def number_of_predictions(self, value):
        assert isinstance(value, int)
        assert value > 0

        self._number_of_predictions = value

    @property
    def values(self):
        return self._values

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
        return np.mean(self.values, axis=1)

    def calculate_confidence_intervale(self, interval):

            p = ((1.0 - interval) / 2.0) * 100
            lower = np.percentile(self.values, p, axis=self.KEY_INT_FOR_SAMPLES_PREDICTIONS)

            p = (interval + ((1.0 - interval) / 2.0)) * 100
            upper = np.percentile(self.values, p, axis=self.KEY_INT_FOR_SAMPLES_PREDICTIONS)

            return lower, upper
