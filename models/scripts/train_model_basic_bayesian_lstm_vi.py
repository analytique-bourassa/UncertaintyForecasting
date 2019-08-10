import matplotlib.pyplot as plt
from models.LSTM_BayesRegressor.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np

from data_generation.data_generators_switcher import DatageneratorsSwitcher

from models.lstm_params import LSTM_parameters
from models.script_parameters.parameters import ExperimentParameters
from models.calibration.diagnostics import calculate_one_sided_cumulative_calibration, calculate_confidence_interval_calibration, calculate_marginal_calibration
from models.LSTM_VI.LSTM_VI_model import LSTM_VI
from data_handling.data_reshaping import reshape_data_for_LSTM, reshape_into_sequences


import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.optim import Adam
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.contrib.autoguide import AutoDiagonalNormal

import torch
import torch.nn as nn


experiment_params = ExperimentParameters()

experiment_params.path = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

path_results = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/"

experiment_params.version = "v0.0.1"
experiment_params.show_figures = False
experiment_params.smoke_test = False
experiment_params.train_lstm = True
experiment_params.save_lstm = False
experiment_params.type_of_data =  "sinus" # options are sin or ar5
experiment_params.name = "feature_extractor_" + experiment_params.type_of_data



# Network params
lstm_params = LSTM_parameters()

if experiment_params.train_lstm is False:
    lstm_params.load("lstm_params_" + experiment_params.name + "_" + experiment_params.version, experiment_params.path)

learning_rate = 3e-3
num_epochs = 250 if not experiment_params.smoke_test else 1

n_data = 3000
dtype = torch.float
length_of_sequences = 10 + 1


######################
# Create data for model
######################

data_generator = DatageneratorsSwitcher(experiment_params.type_of_data)
data = data_generator(n_data)
sequences = reshape_into_sequences(data, length_of_sequences)
all_data = reshape_data_for_LSTM(sequences)

training_data_labels = all_data[length_of_sequences-1, :, 0]

number_of_train_data = floor(0.67*n_data)

data_train = all_data[:, :number_of_train_data, :]

if experiment_params.show_figures:
    plt.plot(all_data[length_of_sequences-2, :, 0])
    plt.plot(training_data_labels)
    plt.ylabel("y (value to forecast)")
    plt.xlabel("time steps")
    plt.legend()
    plt.show()


X_train = data_train[:-1, :, :]
lstm_params.batch_size = X_train.shape[1]

X_train = torch.from_numpy(X_train).type(torch.Tensor)

y_train_numpy = data_train[-1, :, :]
y_train = torch.from_numpy(y_train_numpy).type(torch.Tensor)

lstm_vi = LSTM_VI(lstm_params)
model = lstm_vi.model
guide = AutoDiagonalNormal(model)
optim = Adam({"lr": 0.03})

svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)


num_iterations = 2000
pyro.clear_param_store()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(X_train, y_train)
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(X_train)))

posterior = svi.run(X_train, y_train)

trace_pred = TracePredictive(model,
                             posterior,
                             num_samples=100)

number_of_samples = 100

def get_results( traces, sites):
    get_marginal = lambda traces, sites: EmpiricalMarginal(traces, sites)._get_samples_and_weights()[
        0].detach().cpu().numpy()

    marginal = get_marginal(traces, sites)

    return marginal[:, 0, :].transpose()

post_pred = trace_pred.run(X_train, None)
y = get_results(post_pred, sites=['obs'])

from probabilitic_predictions.probabilistic_predictions_regression import ProbabilisticPredictionsRegression

predictions = ProbabilisticPredictionsRegression()
predictions.number_of_predictions = lstm_params.batch_size
predictions.number_of_samples = number_of_samples
predictions.initialize_to_zeros()

predictions.values = y[0]
predictions.true_values = y_train_numpy


predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)