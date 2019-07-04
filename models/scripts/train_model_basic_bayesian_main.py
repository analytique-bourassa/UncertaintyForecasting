import matplotlib.pyplot as plt
from models.LSTM_BayesRegressor.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np

from models.LSTM_BayesRegressor.gaussian_model_mcmc import GaussianLinearModel_MCMC
from models.LSTM_BayesRegressor.gaussian_model_mcmc_pyro import GaussianLinearModel_MCMC_pyro
from data_generation.data_generators_switcher import DatageneratorsSwitcher

from models.lstm_params import LSTM_parameters
from models.disk_reader_and_writer import save_checkpoint, load_checkpoint
from models.calibration.analysis import show_analysis
from models.script_parameters.parameters import ExperimentParameters
from data_handling.data_reshaping import reshape_data_for_LSTM, reshape_into_sequences

experiment_params = ExperimentParameters()

experiment_params.path = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

experiment_params.version = "v0.0.1"
experiment_params.show_figures = False
experiment_params.smoke_test = False
experiment_params.train_lstm = False
experiment_params.save_lstm = False
experiment_params.type_of_data = "sinus" # options are sin or ar5
experiment_params.name = "feature_extractor_" + experiment_params.type_of_data

assert experiment_params.type_of_data in ["sin", "ar5"]

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

##################################################
# Create and optimize model for feature extraction
##################################################

model = LSTM(lstm_params)
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


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

    load_checkpoint(model, optimizer, experiment_params.path, experiment_params.name + "_" + experiment_params.version)


if experiment_params.save_lstm:
    save_checkpoint(model, optimizer, experiment_params.path, experiment_params.name + "_" + experiment_params.version)
    lstm_params.save(experiment_params.version, experiment_params.path)

y_pred, _ = make_predictions(data_loader, model, all_data, lstm_params.batch_size)
features, y_true = extract_features(data_loader, model, all_data, lstm_params.batch_size)

########################################################
# Create and optimize model for probabilistic predictions
#########################################################

print(np.corrcoef(features.T))
print("mean difference: ", np.mean(np.abs(y_pred[number_of_train_data:] - y_true[number_of_train_data:])))

if experiment_params.show_figures:
    plt.plot(y_pred, label="Preds")
    plt.plot(y_true, label="Data")
    plt.ylabel("y (value to forecast)")
    plt.xlabel("time steps")
    plt.legend()
    plt.show()

    if experiment_params.train_lstm:
        plt.plot(hist)
        plt.show()

X_train, X_test, y_train, y_test = features[:number_of_train_data], features[number_of_train_data:], \
                                   y_true[:number_of_train_data], y_true[number_of_train_data:]


priors_beta, _ = model.last_layers_weights

model_linear_mcmc = GaussianLinearModel_MCMC(X_train, y_train, priors_beta)
model_linear_mcmc.option = "advi"
model_linear_mcmc.sample()
model_linear_mcmc.show_trace()

predictions = model_linear_mcmc.make_predictions(X_test, y_test)

predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)

show_analysis(predictions.values, predictions.true_values, name="LSTM + " + model_linear_mcmc.option)

