import numpy as np
import matplotlib.pyplot as plt


class ProbabilisticPredictions():

    KEY_INT_FOR_SAMPLES_PREDICTIONS = 1
    KEY_INT_FOR_PREDICTIONS_PREDICTIONS = 0
    KEY_INT_TRUE_VALUES = 0

    def __init__(self):

        self._number_of_samples = None
        self._number_of_predictions = None
        self._values = None
        self._true_values = None

    def initialize_to_zeros(self):

        assert self.number_of_predictions is not None
        assert self.number_of_samples is not None

        self.values = np.zeros((self.number_of_predictions, self.number_of_samples))
        self.true_values = np.zeros(self.number_of_predictions)

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

    def calculate_confidence_interval(self, interval):

            p = ((1.0 - interval) / 2.0) * 100
            lower = np.percentile(self.values, p, axis=self.KEY_INT_FOR_SAMPLES_PREDICTIONS)

            p = (interval + ((1.0 - interval) / 2.0)) * 100
            upper = np.percentile(self.values, p, axis=self.KEY_INT_FOR_SAMPLES_PREDICTIONS)

            return lower, upper

    def show_predictions_with_confidence_interval(self, confidence_interval):

        assert 0 <= confidence_interval <= 1.0, "must be between zero and one (including boundary)"

        lower, upper = self.calculate_confidence_interval(confidence_interval)

        x = range(self.number_of_predictions)

        plt.plot(x, self.predictions, label="Preds")
        plt.plot(x, self.true_values, label="Data")
        plt.xlabel("time")
        plt.ylabel("y (value to forecast)")
        plt.title("Prediction with %f confidence interval" % confidence_interval)
        plt.fill_between(x, lower, upper, alpha=0.5)
        plt.legend()
        plt.show()
