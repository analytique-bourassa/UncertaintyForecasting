import matplotlib.pyplot as plt
from models.LSTM_BayesRegressor.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np

from models.LSTM_BayesRegressor.bayesian_linear_regression.bayesian_linear_regression import BayesianLinearModel
from data_generation.data_generators_switcher import DatageneratorsSwitcher

from models.lstm_params import LSTM_parameters
from models.disk_reader_and_writer import save_checkpoint, load_checkpoint
from models.script_parameters.parameters import ExperimentParameters
from models.calibration.diagnostics import calculate_one_sided_cumulative_calibration, calculate_confidence_interval_calibration, calculate_marginal_calibration

from data_handling.data_reshaping import reshape_data_for_LSTM, reshape_into_sequences

def calculate_correlation_score(features):
    return np.linalg.det(np.corrcoef(features.T))

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
experiment_params.type_of_data =  "autoregressive-5" # options are sin or ar5
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

##################################################
# Create and optimize model for feature extraction
##################################################
number_of_experiment_per_type = 100

from models.LSTM_BayesRegressor.experiment_results import ExperimentsResults
results = ExperimentsResults()

from utils.Timer import Timer
import random

for i in range(number_of_experiment_per_type):

    dropout = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ])
    lstm_params.dropout = dropout

    model = LSTM(lstm_params)
    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if experiment_params.train_lstm:
        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            model.hidden = model.init_hidden()

            losses, N_data = make_forward_pass(data_loader, model, loss_fn, data_train, lstm_params.batch_size)
            # if t % 10 == 0:
            # print("Epoch ", t, "MSE: ", losses.item())

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

    for option in BayesianLinearModel.POSSIBLE_OPTION_FOR_POSTERIOR_CALCULATION:

        with Timer("%s number %d " % (option, i)) as timer:


            ########################################################
            # Create and optimize model for probabilistic predictions
            #########################################################

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

            model_linear_mcmc = BayesianLinearModel(X_train, y_train, priors_beta)
            model_linear_mcmc.option = option
            model_linear_mcmc.sample()
            #model_linear_mcmc.show_trace()

            predictions = model_linear_mcmc.make_predictions(X_test, y_test)

            deviation_score_probabilistic_calibration = calculate_confidence_interval_calibration(predictions.values, predictions.true_values)
            deviation_score_exceedance_calibration = calculate_one_sided_cumulative_calibration(predictions.values, predictions.true_values)
            deviation_score_marginal_calibration = calculate_marginal_calibration(predictions.values, predictions.true_values)

            results.confidence_interval_deviance_score.append(deviation_score_probabilistic_calibration)
            results.one_sided_deviance_score.append(deviation_score_exceedance_calibration)
            results.marginal_deviance_score.append(deviation_score_marginal_calibration)

            results.correlation_score.append(calculate_correlation_score(features))
            results.methods.append(option)
            results.elapsed_times.append(timer.elapsed_time())

        #predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)
        #show_analysis(predictions.values, predictions.true_values, name="LSTM + " + model_linear_mcmc.option)

results.save_as_csv(path_results, "results_n_100_ar5")