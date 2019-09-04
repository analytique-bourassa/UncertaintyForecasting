from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from data_generation.data_generators_switcher import DatageneratorsSwitcher
from data_handling.data_reshaping import reshape_data_for_LSTM, reshape_into_sequences

from models.LSTM_CorrelatedDropout.LSTM_not_correlated_dropout import LSTM_not_correlated_dropout
from models.LSTM_CorrelatedDropout.losses import LossRegressionGaussianNoCorrelations

from models.LSTM_CorrelatedDropout.LSTM_correlated_dropout import LSTM_correlated_dropout
from models.LSTM_CorrelatedDropout.losses import LossRegressionGaussianWithCorrelations

from models.model_data_feeder import *

from models.lstm_params import LSTM_parameters
from models.disk_reader_and_writer import save_checkpoint, load_checkpoint
from models.calibration.analysis import show_analysis
from models.script_parameters.parameters import ExperimentParameters
from models.calibration.diagnostics import calculate_one_sided_cumulative_calibration, calculate_confidence_interval_calibration, calculate_marginal_calibration

from models.visualisations import Visualisator
from models.training_tools.early_stopping import EarlyStopping

early_stopper = EarlyStopping(patience=4, verbose=True)

experiment_params = ExperimentParameters()

experiment_params.path = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

path_results = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/"


experiment_params.version = "v0.1.0"
experiment_params.show_figures = False
experiment_params.smoke_test = False
experiment_params.train_lstm = True
experiment_params.save_lstm = True
experiment_params.type_of_data = "autoregressive-5" # options are sin or "autoregressive-5"
experiment_params.name = "feature_extractor_" + experiment_params.type_of_data



# Network params
lstm_params = LSTM_parameters()
lstm_params.batch_size = 1
lstm_params.hidden_dim = 5

if experiment_params.train_lstm is False:
    lstm_params.load("lstm_params_" + experiment_params.name + "_" + experiment_params.version, experiment_params.path)

learning_rate = 1e-3
num_epochs = 2300 if not experiment_params.smoke_test else 1

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

lstm_params.dropout = 0.4

model = LSTM_correlated_dropout(lstm_params)
model.cuda()

loss_fn = LossRegressionGaussianWithCorrelations(0.1, lstm_params.hidden_dim)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)



if experiment_params.train_lstm:
    hist = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):
        model.hidden = model.init_hidden()

        losses, N_data = make_forward_pass_output_specific(data_loader_sequences, model, loss_fn, data_train, lstm_params.batch_size)

        hist[epoch] = losses

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

y_pred, _ = make_predictions(data_loader_sequences, model, all_data, lstm_params.batch_size)
features, y_true = extract_features(data_loader_sequences, model, all_data, lstm_params.batch_size)
