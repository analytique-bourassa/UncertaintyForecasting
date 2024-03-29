import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

from uncertainty_forecasting.utils.validator import Validator

DEFAULT_TICKS_SIZE = 19
DEFAULT_LABEL_SIZE = 22
DEFAULT_LINEWIDTH = 3.0
DEFAULT_TITLE_SIZE = 26
DEFAULT_LEGEND_SIZE = 19
DEFAULT_TRANSPARENCY = 0.5
DEFAULT_MARKER_SIZE = 6.0

from matplotlib import rcParams
rcParams['axes.titlepad'] = 20

class Visualisator():

    @staticmethod
    def show_time_series(data, title):

        plt.title(title, size=DEFAULT_TITLE_SIZE)
        plt.plot(data, linewidth=DEFAULT_LINEWIDTH)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        plt.ylabel("y", size=DEFAULT_LABEL_SIZE, rotation=0)
        plt.xlabel("Time", size=DEFAULT_LABEL_SIZE)

        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()

    @staticmethod
    def show_distribution(data, title,name, normal_fit=True):

        if normal_fit:
            x = np.linspace(min(data), max(data), 100)
            y = norm.pdf(x, loc=data.mean(), scale=data.std())
            plt.plot(x, y, label="normal fit", linewidth=DEFAULT_LINEWIDTH)

        plt.title(title, size=DEFAULT_TITLE_SIZE)
        plt.ylabel("Density", size=DEFAULT_LABEL_SIZE)
        plt.xlabel(name, size=DEFAULT_LABEL_SIZE)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        sns.distplot(data,
                     norm_hist=True,
                     label="density of %s" % name, kde=False)

        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()

    @staticmethod
    def show_epoch_convergence(data, title, name, number_of_burned_step=200):

        n_epochs = data.shape[0]
        plt.title(title, size=DEFAULT_TITLE_SIZE)
        plt.plot(range(number_of_burned_step, n_epochs), data[number_of_burned_step:])

        plt.ylabel(name, size=DEFAULT_LABEL_SIZE, rotation=0)
        plt.xlabel("epoch", size=DEFAULT_LABEL_SIZE)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.show()

    @staticmethod
    def show_predictions(y_pred, y_true, dataset="training"):

        plt.plot(y_pred, label="predictions", linewidth=DEFAULT_LINEWIDTH)
        plt.plot(y_true, label="true values", linewidth=DEFAULT_LINEWIDTH)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        plt.ylabel("y", size=DEFAULT_LABEL_SIZE, rotation=0)
        plt.xlabel("Time", size=DEFAULT_LABEL_SIZE)

        plt.title("Comparing predictions with true values on %s set" % dataset, size=DEFAULT_TITLE_SIZE)
        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()

    @staticmethod
    def show_predictions_with_confidence_interval(predictions,
                                                  true_values,
                                                  lower_bound,
                                                  upper_bound,
                                                  confidence_interval):

        Validator.check_value_is_between_zero_and_one_inclusive(confidence_interval)

        number_of_predictions = len(predictions)
        x = range(number_of_predictions)

        plt.plot(x, predictions, label="predictions", linewidth=DEFAULT_LINEWIDTH)
        plt.plot(x, true_values, label="true values", linewidth=DEFAULT_LINEWIDTH)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        plt.xlabel("Time", size=DEFAULT_LABEL_SIZE)
        plt.ylabel("y (value to forecast)", size=DEFAULT_LABEL_SIZE)
        plt.title("Prediction with %2.0f %% confidence interval" % (100.0*confidence_interval),
                  size=DEFAULT_TITLE_SIZE)

        plt.fill_between(x, lower_bound, upper_bound, alpha=DEFAULT_TRANSPARENCY)
        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()

    @staticmethod
    def show_predictions_and_training_data_with_confidence_interval(predictions,
                                                                    train_data,
                                                                    true_values,
                                                                    lower_bound,
                                                                    upper_bound,
                                                                    confidence_interval):

        Validator.check_value_is_between_zero_and_one_inclusive(confidence_interval)

        n_predictions = len(predictions)
        n_training_data = len(train_data)
        n_total = n_training_data + n_predictions

        x_for_test = range(n_training_data, n_training_data + n_predictions)
        x_total = range(n_total)

        data = np.zeros(n_total)
        data[:n_training_data] = train_data.flatten()
        data[n_training_data:] = true_values.flatten()

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        plt.plot(x_for_test, predictions, label="predictions", linewidth=DEFAULT_LINEWIDTH)
        plt.plot(x_total, data, label="true values", linewidth=DEFAULT_LINEWIDTH)

        plt.xlabel("Time", size=DEFAULT_LABEL_SIZE)
        plt.ylabel("y (value to forecast)", size=DEFAULT_LABEL_SIZE)
        plt.title("Prediction with %2.0f %% confidence interval" % 100 * confidence_interval,
                  size=DEFAULT_TITLE_SIZE)

        plt.fill_between(x, lower_bound, upper_bound, alpha=DEFAULT_TRANSPARENCY)
        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()


    @staticmethod
    def show_multiple_distribution(values_per_distribution, labels, title, variable_name):

        Validator.check_type(values_per_distribution, np.ndarray)
        Validator.check_all_elements_type(labels, str)
        Validator.check_type(title, str)
        Validator.check_type(variable_name, str)

        number_of_labels = len(labels)
        number_of_distribution = values_per_distribution.shape[0]
        Validator.check_matching_dimensions(number_of_distribution, number_of_labels)

        for index_label, label in enumerate(labels):

            values = values_per_distribution[index_label]
            sns.distplot(values, hist=False, rug=False, label=label)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        plt.xlabel(variable_name, size=DEFAULT_LABEL_SIZE)
        plt.ylabel("Density", size=DEFAULT_LABEL_SIZE)
        plt.title("Distributions",
                  size=DEFAULT_TITLE_SIZE)

        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()

    @staticmethod
    def show_calibration_curves(x_values, values_for_curves, labels, title_suffix):

        Validator.check_type(values_for_curves, np.ndarray)
        Validator.check_type(x_values, np.ndarray)
        Validator.check_all_elements_type(labels, str)
        Validator.check_type(title_suffix, str)

        number_of_labels = len(labels)
        number_of_curves = values_for_curves.shape[0]
        Validator.check_matching_dimensions(number_of_curves, number_of_labels)

        for index_label, label in enumerate(labels):
            values = values_for_curves[index_label]
            plt.plot(x_values, values, ".-", label=label, markersize=DEFAULT_MARKER_SIZE)

        plt.plot(x_values, x_values)

        plt.xticks(fontsize=DEFAULT_TICKS_SIZE)
        plt.yticks(fontsize=DEFAULT_TICKS_SIZE)

        plt.xlabel("p (confidence of prediction)", size=DEFAULT_LABEL_SIZE)
        plt.ylabel("accuracy", size=DEFAULT_LABEL_SIZE)

        title = "Calibation curve" if number_of_curves == 1 else "Calibation curves"
        plt.title(title + " " + title_suffix,
                  size=DEFAULT_TITLE_SIZE)

        plt.legend(fontsize=DEFAULT_LEGEND_SIZE)
        plt.show()

