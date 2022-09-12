from scipy.special import erfinv
from math import sqrt, floor
import numpy as np
import matplotlib.pyplot as plt

N_INTERVALS = 101
n_possible_values = 100

# Last index for bins
INDEX_CORRECT_VALUE = 0
INDEX_TOTAL_VALUES_IN_BIN = 1

def in_interval(y_mean, y_true, sigma):

    if y_true >= y_mean - sigma and y_true <= y_mean + sigma:
        return True
    else:
        return False

def calculate_lower_upper_confidence_interval(random_samples, alpha):

    p = ((1.0 - alpha) / 2.0) * 100


    lower =  np.percentile(random_samples, p)

    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = np.percentile(random_samples, p)

    return lower, upper

def calculate_confidence_interval_calibration(y_pred, y_true, show=False):

    intervals = np.linspace(0., 1, N_INTERVALS)
    n_time_steps = y_pred.shape[0]

    counts = np.zeros(N_INTERVALS)

    for index_interval, interval in enumerate(intervals):

        for index_y, y_true_value in enumerate(y_true):

            samples_y_pred = y_pred[index_y]

            lower, upper = calculate_lower_upper_confidence_interval(samples_y_pred, interval)

            if lower <= y_true_value <= upper:
                counts[index_interval] += 1

    counts /= n_time_steps
    counts[0] = 0
    counts[-1] = 1.0

    if show:
        plt.plot(intervals, counts)
        plt.plot(intervals, intervals, "-")
        plt.xlabel("Confidence intervals")
        plt.ylabel("frequency ratio in confidence interval")
        plt.title("Confidence interval calibration")
        plt.show()

    deviations = np.abs(counts - intervals)
    deviation_score_probabilistic_calibration = (intervals[1] - intervals[0])*deviations.sum()

    return deviation_score_probabilistic_calibration

def show_in_intervals_guassian_forecast(y_pred, y_true):


    y_mean = np.mean(y_pred, axis=1)
    error = np.std(y_pred, axis=1)

    intervals = np.linspace(0., 1, N_INTERVALS)

    counts = np.zeros(N_INTERVALS)

    for index_interval, interval in enumerate(intervals):

        for index_y, y_mean_value in enumerate(y_mean):

            factor = sqrt(2) * erfinv(interval)
            sigma = factor * error[index_y]

            if in_interval(y_mean_value, y_true[index_y], sigma):
                counts[index_interval] += 1

    counts /= y_mean.shape[0]

    plt.plot(intervals, counts)
    plt.plot(intervals, intervals, "-")
    plt.xlabel("Confidence intervals")
    plt.ylabel("frequency ratio in confidence interval")
    plt.title("Confidence interval calibration")
    plt.show()

    deviation_score_probabilistic_calibration = (intervals[1] - intervals[0])*np.abs(counts - intervals).sum()
    return deviation_score_probabilistic_calibration

def calculate_one_sided_cumulative_calibration(y_pred, y_true, show=False):

    y_mean = np.mean(y_pred, axis=1)
    error = np.std(y_pred, axis=1)

    n_values = N_INTERVALS
    p_values = np.linspace(0., 1, n_values)

    def calculate_empirical_cdf(y_mean, y_true, error, p_values):

        cdf_calculated = np.zeros(p_values.shape[0])
        cdf_limits = sqrt(2) * np.vectorize(erfinv)(2 * p_values - 1)

        for index_cdf, cdf_limit in enumerate(cdf_limits):
            for index_value, y_true_value in enumerate(y_true):
                if y_true_value <= y_mean[index_value] + error[index_value] * cdf_limit:
                    cdf_calculated[index_cdf] += 1

        cdf_calculated /= y_true.shape[0]
        relative_frequency = cdf_calculated[1:] / p_values[1:]

        return cdf_calculated, relative_frequency

    cdf_empirical, relative_frequency = calculate_empirical_cdf(y_mean, y_true, error, p_values)

    if show:
        plt.subplot(121)

        plt.plot(p_values, cdf_empirical)
        plt.plot(p_values, p_values, "-")
        plt.xlabel("p")
        plt.ylabel("Cumulative distribution function estimate")
        plt.title("Calibration of the cumulative distribution function")

        plt.subplot(122)
        plt.bar(p_values[1:], relative_frequency, align='center', width=0.02, alpha=0.7)
        plt.plot(p_values, np.ones(p_values.shape[0]), "-")
        plt.xlabel("p")
        plt.ylabel("relative frequency")
        plt.title("Calibration of the cumulative distribution function")
        plt.show()

    deviation_score_exceedance_calibration = (p_values[1] - p_values[0])*np.abs(cdf_empirical - p_values).sum()
    return deviation_score_exceedance_calibration



def calculate_marginal_calibration(y_pred, y_true, show=False):

    min_y = min(y_pred.min(), y_true.min())
    max_y = max(y_pred.max(), y_true.max())

    y_mean = np.mean(y_pred, axis=1)

    y_values = np.linspace(min_y, max_y, n_possible_values)

    n_predictions = y_mean.shape[0]
    cdf_forecast = np.zeros(n_possible_values)
    cdf_true = np.zeros(n_possible_values)

    for index_y, y_value in enumerate(y_values):

        cdf_forecast[index_y] = sum(1*(y_mean <= y_value*np.ones(n_predictions)))/n_predictions
        cdf_true[index_y] = sum(1 * (y_true <= y_value * np.ones(n_predictions))) / n_predictions

    if show:
        plt.xlabel("CDF of observations")
        plt.ylabel("CDF of  forecasts")
        p_values = np.linspace(0,1,n_possible_values+1)

        plt.plot(cdf_true, cdf_forecast)
        plt.plot(p_values, p_values, "--")

        plt.title("Marginal calibration")
        plt.show()

    deviation_score_marginal_calibration = ((cdf_true[1:] - cdf_true[:-1]) * np.abs(cdf_true[1:] - cdf_forecast[1:])).sum()
    return deviation_score_marginal_calibration


def calculate_static_calibration_error(y_pred, y_true, confidences, number_of_classes):

    n_bins = 8
    p_values = np.linspace(0., 1, n_bins + 1)

    index_second_value = 1
    index_first_value = 0
    interval = p_values[index_second_value] - p_values[index_first_value]

    bins = seperate_probabilistic_classification_into_bins(confidences, y_true, n_bins, number_of_classes, interval)
    accuracy_per_bin, curve = calculate_accuracy_per_bins(bins, n_bins, number_of_classes)
    means_per_bin = 0.5*(p_values[1:] + p_values[:-1])
    deviation_score = calculate_deviation_score(bins, accuracy_per_bin, means_per_bin, number_of_classes)

    return curve, means_per_bin, deviation_score


def seperate_probabilistic_classification_into_bins(confidences,y_true, n_bins, number_of_classes, interval):

    bins = np.zeros((n_bins, number_of_classes, 2)) # first in true, second is total put in class
    for index_instance in range(y_true.shape[0]):
        for index_class in range(number_of_classes):

            confidence_instance_specific_class = confidences[index_instance, index_class]

            bin_index = int(floor(confidence_instance_specific_class/interval))
            bin_index = min(n_bins - 1, bin_index)
            bin_index = max(0, bin_index)

            true_label = y_true[index_instance]

            bins[bin_index, index_class, INDEX_TOTAL_VALUES_IN_BIN] += 1
            bins[bin_index, index_class, INDEX_CORRECT_VALUE] += 1*(index_class == true_label)

    return bins

def calculate_accuracy_per_bins(bins, n_bins, number_of_classes):

    accuracy_per_bin = np.zeros((n_bins, number_of_classes))
    curve = np.zeros(n_bins)

    for bin_index in range(n_bins):

        curve[bin_index] = bins[bin_index, :, INDEX_CORRECT_VALUE].sum() / max(
            bins[bin_index, :, INDEX_TOTAL_VALUES_IN_BIN].sum(), 1)

        for class_index in range(number_of_classes):
            accuracy_per_bin[bin_index, class_index] = bins[bin_index, class_index, INDEX_CORRECT_VALUE]
            accuracy_per_bin[bin_index, class_index] /= max(bins[bin_index, class_index, INDEX_TOTAL_VALUES_IN_BIN], 1)

    return accuracy_per_bin, curve


def calculate_deviation_score(bins, accuracy_per_bin, means_per_bin, number_of_classes):

    deviation_score = 0.0
    for class_index in range(number_of_classes):
        vector_deviations = np.abs(accuracy_per_bin[:, class_index] - means_per_bin) * bins[:, class_index,
                                                                                       INDEX_TOTAL_VALUES_IN_BIN]
        deviation_score += vector_deviations.sum()

    n_total_putted_in_bins = bins[:, :, INDEX_TOTAL_VALUES_IN_BIN].sum()
    deviation_score /= number_of_classes * n_total_putted_in_bins

    return deviation_score

