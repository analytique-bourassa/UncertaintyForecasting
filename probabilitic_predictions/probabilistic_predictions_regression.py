import numpy as np
import matplotlib.pyplot as plt

from models.visualisations import Visualisator

DEFAULT_LEGEND_SIZE = 19

class ProbabilisticPredictionsRegression():

    KEY_INT_FOR_SAMPLES_PREDICTIONS = 1
    KEY_INT_FOR_PREDICTIONS_PREDICTIONS = 0
    KEY_INT_TRUE_VALUES = 0
    KEY_FOR_NUMBER_OF_TRAINING_DATA = 0

    def __init__(self):

        self._number_of_samples = None
        self._number_of_predictions = None
        self._values = None
        self._true_values = None

        self._train_data = None


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

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, value):
        assert isinstance(value, np.ndarray)

        self._train_data = value

    def calculate_confidence_interval(self, interval):

            p = ((1.0 - interval) / 2.0) * 100
            lower = np.percentile(self.values, p, axis=self.KEY_INT_FOR_SAMPLES_PREDICTIONS)

            p = (interval + ((1.0 - interval) / 2.0)) * 100
            upper = np.percentile(self.values, p, axis=self.KEY_INT_FOR_SAMPLES_PREDICTIONS)

            return lower, upper

    def show_predictions_with_confidence_interval(self, confidence_interval):

        assert 0 <= confidence_interval <= 1.0, "must be between zero and one (including boundary)"

        lower, upper = self.calculate_confidence_interval(confidence_interval)

        Visualisator.show_predictions_with_confidence_interval(self.predictions,
                                                               self.true_values,
                                                               lower,
                                                               upper,
                                                               confidence_interval)

    def show_predictions_with_training_data(self, confidence_interval):

        assert 0 <= confidence_interval <= 1.0, "must be between zero and one (including boundary)"
        assert self.train_data is not None, "must have train data"

        lower, upper = self.calculate_confidence_interval(confidence_interval)

        n_training_data = self._train_data.shape[self.KEY_FOR_NUMBER_OF_TRAINING_DATA]
        n_total = n_training_data + self.number_of_predictions

        x_for_test = range(n_training_data, n_training_data + self.number_of_predictions)
        x_total = range(n_total)

        data = np.zeros(n_total)
        data[:n_training_data] = self.train_data.flatten()
        data[n_training_data:] = self.true_values.flatten()

        plt.plot(x_for_test, self.predictions, label="predictions", linewidth=3.0)
        plt.plot(x_total, data, label="true values", linewidth=3.0)

        plt.xlabel("Time", size=22)
        plt.ylabel("y", size=22, rotation=0)

        plt.title("Predictions with %2.0f %% confidence interval" % (100*confidence_interval), size=26)
        plt.fill_between(x_for_test, lower, upper, alpha=0.5)

        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()
