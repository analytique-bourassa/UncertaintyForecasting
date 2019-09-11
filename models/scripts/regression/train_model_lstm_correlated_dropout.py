from tqdm import tqdm
import itertools

from data_generation.data_generators_switcher import DatageneratorsSwitcher
from data_handling.data_reshaping import reshape_data_for_LSTM, reshape_into_sequences

from models.regression.LSTM_CorrelatedDropout.LSTM_not_correlated_dropout import LSTM_not_correlated_dropout
from models.regression.LSTM_CorrelatedDropout.losses import LossRegressionGaussianNoCorrelations

from models.model_data_feeder import *

from models.lstm_params import LSTM_parameters
from models.disk_reader_and_writer import save_checkpoint, load_checkpoint
from models.script_parameters.parameters import ExperimentParameters

from models.calibration.diagnostics import calculate_one_sided_cumulative_calibration, calculate_confidence_interval_calibration, calculate_marginal_calibration
from models.calibration.analysis import show_analysis

from probabilitic_predictions.probabilistic_predictions_regression import ProbabilisticPredictionsRegression
from visualisations.visualisations import Visualisator
from models.training_tools.early_stopping import EarlyStopping

early_stopper = EarlyStopping(patience=4, verbose=True)

experiment_params = ExperimentParameters()

experiment_params.path = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

path_results = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/"


experiment_params.version = "v0.1.0"
experiment_params.show_figures = True
experiment_params.smoke_test = False
experiment_params.train_lstm = True
experiment_params.save_lstm = False
experiment_params.type_of_data = "autoregressive-5" # options are sin or "autoregressive-5"
experiment_params.name = "feature_extractor_" + experiment_params.type_of_data



# Network params
lstm_params = LSTM_parameters()
lstm_params.batch_size = 20
lstm_params.hidden_dim = 5

if experiment_params.train_lstm is False:
    lstm_params.load("lstm_params_" + experiment_params.name + "_" + experiment_params.version, experiment_params.path)

learning_rate = 1e-3
num_epochs = 20 if not experiment_params.smoke_test else 1
num_epochs_pretraining = 500 if not experiment_params.smoke_test else 1

n_data = 1000
length_of_sequences = 7 + 1


######################
# Create data for model
######################

data_generator = DatageneratorsSwitcher(experiment_params.type_of_data)
data = data_generator(n_data)

sequences = reshape_into_sequences(data, length_of_sequences)
all_data = reshape_data_for_LSTM(sequences)

training_data_labels = all_data[length_of_sequences-1, :, 0]

number_of_train_data = floor(0.5*n_data)
val_set_end = floor(0.7*n_data)

data_train = all_data[:, :number_of_train_data, :]
data_validation = all_data[:, number_of_train_data:val_set_end, :]

if experiment_params.show_figures:
    Visualisator.show_time_series(data=all_data[length_of_sequences-2, :, 0],
                                  title="Data generated using an ARMA(4,2) process")


##################################################
# Create and optimize model for feature extraction
##################################################

lstm_params.dropout = 0.0

#model = LSTM_correlated_dropout(lstm_params)
model = LSTM_not_correlated_dropout(lstm_params)
model.cuda()

model.show_summary()


#loss_fn = LossRegressionGaussianWithCorrelations(0.1, lstm_params.hidden_dim)
loss_fn = LossRegressionGaussianNoCorrelations(1.0, lstm_params.hidden_dim)

params = list(model.parameters()) + list(model.prediction_sigma)

optimizer_1 = torch.optim.Adam(itertools.chain(model.parameters(),[model.weights_mu]),
                             lr=learning_rate)

optimizer_2 = torch.optim.Adam(itertools.chain( [model.weights_mu,
                                                                   model.prediction_sigma,
                                                                  model.covariance_vector]),
                             lr=5e-3)

loss_function_pretraining = torch.nn.MSELoss(size_average=False)

if experiment_params.train_lstm:

    hist_1 = np.zeros(num_epochs_pretraining)
    for epoch in tqdm(range(num_epochs_pretraining)):
        model.hidden = model.init_hidden()

        losses, N_data = make_forward_pass(data_loader_sequences, model, loss_function_pretraining, data_train,
                                                           lstm_params.batch_size)

        hist_1[epoch] = losses

        optimizer_1.zero_grad()
        losses.backward()
        optimizer_1.step()

    model.is_pretraining = False
    model.show_summary()

    hist = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):
        model.hidden = model.init_hidden()

        losses, N_data = make_forward_pass_output_specific(data_loader_sequences, model, loss_fn, data_train, lstm_params.batch_size)

        hist[epoch] = losses

        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()

else:

    load_checkpoint(model, optimizer_2, experiment_params.path,
                    experiment_params.name + "_" + experiment_params.version)

if experiment_params.save_lstm:
    save_checkpoint(model, optimizer_2, experiment_params.path,
                    experiment_params.name + "_" + experiment_params.version)
    lstm_params.save(experiment_params.version, experiment_params.path)

y_pred, y_true = make_predictions(data_loader_sequences, model, all_data, lstm_params.batch_size)

number_of_train_data = floor(0.7*n_data)
y_train, y_test = y_true[:number_of_train_data], y_true[number_of_train_data:]

model.show_summary()

if experiment_params.show_figures:

    if experiment_params.train_lstm:

        Visualisator.show_epoch_convergence(data=hist_1,
                                            name="Loss",
                                            title="Loss convergence curve for the LSTM training",
                                            number_of_burned_step=0)

        Visualisator.show_epoch_convergence(data=hist,
                                            name="Loss",
                                            title="Loss convergence curve for the LSTM training",
                                            number_of_burned_step=0)


predictions = ProbabilisticPredictionsRegression()
predictions.number_of_predictions = y_pred.shape[0]
predictions.number_of_samples = y_pred.shape[1]
predictions.initialize_to_zeros()

predictions.values = y_pred
predictions.true_values = y_true
predictions.show_predictions_with_confidence_interval(0.95)

deviation_score_probabilistic_calibration = calculate_confidence_interval_calibration(predictions.values, predictions.true_values)
deviation_score_exceedance_calibration = calculate_one_sided_cumulative_calibration(predictions.values, predictions.true_values)
deviation_score_marginal_calibration = calculate_marginal_calibration(predictions.values, predictions.true_values)

predictions.train_data = y_train
predictions.show_predictions_with_training_data(confidence_interval=0.95)

show_analysis(predictions.values, predictions.true_values, name="LSTM + dropout")


