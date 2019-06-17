from scipy.special import erfinv
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

def in_interval(y_mean, y_true, sigma):

    if y_true >= y_mean - sigma and y_true <= y_mean + sigma:
        return True
    else:
        return False

def show_in_intervals(y_pred, y_true):


    y_mean = np.mean(y_pred, axis=1)
    error = np.std(y_pred, axis=1)

    n_intervals = 21
    intervals = np.linspace(0., 1, n_intervals)

    counts = np.zeros(n_intervals)

    for index_interval, interval in enumerate(intervals):

        for index_y, y_mean_value in enumerate(y_mean):

            factor = sqrt(2) * erfinv(interval)
            sigma = factor * error[index_y]

            if in_interval(y_mean_value, y_true[index_y], sigma):
                counts[index_interval] += 1

    counts /= y_mean.shape[0]

    plt.plot(intervals, counts)
    plt.plot(intervals, intervals, "-")
    plt.xlabel("confidence intervale")
    plt.ylabel("frequency ratio in confidence interval")
    plt.title("calibration")
    plt.show()

    deviation_score_probabilistic_calibration = (intervals[1] - intervals[0])*np.abs(counts - intervals).sum()
    return deviation_score_probabilistic_calibration

def show_empirical_cdf(y_pred, y_true):

    y_mean = np.mean(y_pred, axis=1)
    error = np.std(y_pred, axis=1)

    n_values = 21
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

    plt.subplot(121)

    plt.plot(p_values, cdf_empirical)
    plt.plot(p_values, p_values, "-")
    plt.xlabel("p")
    plt.ylabel("cdf empirical")
    plt.title("calibration  cdf")

    plt.subplot(122)
    plt.bar(p_values[1:], relative_frequency, align='center', width=0.02, alpha=0.7)
    plt.plot(p_values, np.ones(p_values.shape[0]), "-")
    plt.xlabel("p")
    plt.ylabel("relative frequency")
    plt.title("calibration  cdf")
    plt.show()

    deviation_score_exceedance_calibration = (p_values[1] - p_values[0]) * np.abs(cdf_empirical - p_values).sum()
    return deviation_score_exceedance_calibration



def show_marginal_calibration(y_pred, y_true):

    min_y = min(y_pred.min(), y_true.min())
    max_y = max(y_pred.max(), y_true.max())

    y_mean = np.mean(y_pred, axis=1)

    n_possible_values = 100
    y_values = np.linspace(min_y, max_y, n_possible_values)

    n_predictions = y_mean.shape[0]
    cdf_forecast = np.zeros(n_possible_values)
    cdf_true = np.zeros(n_possible_values)

    for index_y, y_value in enumerate(y_values):

        cdf_forecast[index_y] = sum(1*(y_mean <= y_value*np.ones(n_predictions)))/n_predictions
        cdf_true[index_y] = sum(1 * (y_true <= y_value * np.ones(n_predictions))) / n_predictions

    plt.xlabel("cdf true")
    plt.ylabel("cdf forcecasts")
    p_values = np.linspace(0,1,n_possible_values+1)

    plt.plot(cdf_true, cdf_forecast)
    plt.plot(p_values, p_values, "--")

    plt.title("marginal calibration")
    plt.show()

    deviation_score_marginal_calibration = ((cdf_true[1:] - cdf_true[:-1]) * np.abs(cdf_true[1:] - cdf_forecast[1:])).sum()
    return deviation_score_marginal_calibration


def empiral_confidance_interval():

    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(means_of_random_samples, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(means_of_random_samples, p))
    print('%.1f confidence interval %.4f and %.4f' % (alpha * 100, lower, upper))