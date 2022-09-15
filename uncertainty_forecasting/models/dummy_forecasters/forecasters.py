import numpy as np

NOISE_MEAN = 0
NOISE_STD = 0.1
KEY_N_DATA = 0
BIAS = 0.1
n_samples = 100


def generate_ideal_forecaster_values(mean_values):

    n_data = mean_values.shape[KEY_N_DATA]
    ideal_forecaster = np.zeros((n_samples, n_data))

    for index_sample in range(n_samples):
        ideal_forecaster[index_sample] = mean_values + np.random.normal(NOISE_MEAN, NOISE_STD, n_data)

    return ideal_forecaster


def generate_unfocused_forecaster_values(mean_values):

    n_data = mean_values.shape[KEY_N_DATA]
    unfocused_forecaster = np.zeros((n_samples, n_data))

    for index_sample in range(n_samples):
        unfocused_forecaster[index_sample] = mean_values + BIAS*(2*np.random.randint(0, 2, size=(n_data))-1)
        unfocused_forecaster[index_sample] += np.random.normal(NOISE_MEAN, NOISE_STD, n_data)

    return unfocused_forecaster


def generate_asymetric_bias_forecaster_values(mean_values):

    n_data = mean_values.shape[KEY_N_DATA]
    asymetric_bias_forecaster = np.zeros((n_samples, n_data))

    for index_sample in range(n_samples):
        asymetric_bias_forecaster[index_sample] = mean_values
        asymetric_bias_forecaster[index_sample] += BIAS * np.sign(asymetric_bias_forecaster[index_sample])
        asymetric_bias_forecaster[index_sample] += np.random.normal(NOISE_MEAN, NOISE_STD, n_data)

    return asymetric_bias_forecaster

