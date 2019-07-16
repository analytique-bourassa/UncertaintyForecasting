import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from data_generation.data_generators_switcher import DatageneratorsSwitcher
from data_handling.data_reshaping import reshape_data_for_LSTM, reshape_into_sequences
from models.LSTM_BayesRegressor.LSTM import LSTM
from models.model_data_feeder import *
from models.LSTM_BayesRegressor.gaussian_model_mcmc import GaussianLinearModel_MCMC
from models.lstm_params import LSTM_parameters
from models.disk_reader_and_writer import save_checkpoint, load_checkpoint
from models.calibration.analysis import show_analysis
from models.script_parameters.parameters import ExperimentParameters
from models.calibration.diagnostics import calculate_one_sided_cumulative_calibration, calculate_confidence_interval_calibration, calculate_marginal_calibration


experiment_params = ExperimentParameters()

experiment_params.path = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

path_results = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/"


experiment_params.version = "v0.0.3"
experiment_params.show_figures = True
experiment_params.smoke_test = False
experiment_params.train_lstm = True
experiment_params.save_lstm = True
experiment_params.type_of_data = "autoregressive-5" # options are sin or "autoregressive-5"
experiment_params.name = "feature_extractor_" + experiment_params.type_of_data



# Network params
lstm_params = LSTM_parameters()
lstm_params.batch_size = 10
lstm_params.hidden_dim = 5

if experiment_params.train_lstm is False:
    lstm_params.load("lstm_params_" + experiment_params.name + "_" + experiment_params.version, experiment_params.path)

learning_rate = 1e-3
num_epochs = 3000 if not experiment_params.smoke_test else 1

n_data = 1000
length_of_sequences = 10 + 1


######################
# Create data for model
######################

data_generator = DatageneratorsSwitcher(experiment_params.type_of_data)
data = data_generator(n_data)

sequences = reshape_into_sequences(data, length_of_sequences)
all_data = reshape_data_for_LSTM(sequences)

training_data_labels = all_data[length_of_sequences-1, :, 0]

number_of_train_data = floor(0.6*n_data)

data_train = all_data[:, :number_of_train_data, :]

if experiment_params.show_figures:
    plt.title("Data generated using an autoregressive process of order 5", size=26)
    plt.plot(all_data[length_of_sequences-2, :, 0], linewidth=3.0)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.ylabel("y", size=22, rotation=0)
    plt.xlabel("Time", size=22)
    plt.legend()
    plt.show()

##################################################
# Create and optimize model for feature extraction
##################################################

lstm_params.dropout = 0.4

model = LSTM(lstm_params)
model.cuda()

loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)

if experiment_params.train_lstm:
    hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        model.hidden = model.init_hidden()

        losses, N_data = make_forward_pass(data_loader, model, loss_fn, data_train, lstm_params.batch_size)
        if t % 10 == 0:
            print("Epoch ", t, "MSE: ", losses.item())

        hist[t] = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
else:

    load_checkpoint(model, optimizer, experiment_params.path,
                    experiment_params.name + "_" + experiment_params.version)

if experiment_params.save_lstm:
    save_checkpoint(model, optimizer, experiment_params.path,
                    experiment_params.name + "_" + experiment_params.version)
    lstm_params.save(experiment_params.version, experiment_params.path)

y_pred, _ = make_predictions(data_loader, model, all_data, lstm_params.batch_size)
features, y_true = extract_features(data_loader, model, all_data, lstm_params.batch_size)

errors = y_true.flatten() - y_pred.flatten()
x = np.linspace(min(errors), max(errors), 100)
y = norm.pdf(x, loc=errors.mean(), scale=errors.std())

plt.plot(x, y, label="normal fit", linewidth=3.0)
plt.title("Distribution of errors ($y_{true} - y_{predicted}$)", size=26)
plt.ylabel("Probability", size=22)
plt.xlabel("error", size=22)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)

sns.distplot(errors, norm_hist=True, label="histogram of errors")
plt.legend()
plt.show()

########################################################
# Create and optimize model for probabilistic predictions
#########################################################

if experiment_params.show_figures:

    plt.plot(y_pred, label="predictions", linewidth=3.0)
    plt.plot(y_true, label="true value", linewidth=3.0)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.ylabel("y", size=22, rotation=0)
    plt.xlabel("Time", size=22)
    plt.title("Comparing prediction with true data on training set")
    plt.legend()
    plt.show()

    if experiment_params.train_lstm:
        burn_loss = 200
        plt.title("Convergence curve of the loss for the training of the LSTM", size=14)
        plt.plot(hist[burn_loss:])
        plt.ylabel("Loss", size=12)
        plt.xlabel("epoch", size=12)
        plt.show()

X_train, X_test, y_train, y_test = features[:number_of_train_data], features[number_of_train_data:], \
                                   y_true[:number_of_train_data], y_true[number_of_train_data:]

model.cpu()
priors_beta, _ = model.last_layers_weights

model_linear_mcmc = GaussianLinearModel_MCMC(X_train, y_train, priors_beta)
model_linear_mcmc.option = "Hybrid"
model_linear_mcmc.sample()
model_linear_mcmc.show_trace()

predictions = model_linear_mcmc.make_predictions(X_test, y_test)

deviation_score_probabilistic_calibration = calculate_confidence_interval_calibration(predictions.values, predictions.true_values)
deviation_score_exceedance_calibration = calculate_one_sided_cumulative_calibration(predictions.values, predictions.true_values)
deviation_score_marginal_calibration = calculate_marginal_calibration(predictions.values, predictions.true_values)

predictions.train_data = y_train
predictions.show_predictions_with_training_data(confidence_interval=0.95)

show_analysis(predictions.values, predictions.true_values, name="LSTM + " + model_linear_mcmc.option)


