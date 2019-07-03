import matplotlib.pyplot as plt
from models.LSTM_BayesRegressor.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np
from models.LSTM_BayesRegressor.gaussian_model_mcmc import GaussianLinearModel_MCMC
from models.LSTM_BayesRegressor.gaussian_model_mcmc_pyro import GaussianLinearModel_MCMC_pyro
from models.lstm_params import LSTM_parameters
from models.disk_reader_and_writer import save_checkpoint, load_checkpoint

from models.calibration.analysis import show_analysis

from data_generation.data_generator import return_arma_data


PATH = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

VERSION = "v0.0.1"
SHOW_FIGURES = False
SMOKE_TEST = False
TRAIN_LSTM = False
SAVE_LSTM = False
TYPE_OF_DATA = "sin" # options are sin or ar5
NAME = "feature_extractor_sin"

assert TYPE_OF_DATA in ["sin", "ar5"]

# Network params
lstm_params = LSTM_parameters()
if TRAIN_LSTM is False:
    lstm_params.load("lstm_params_" + NAME + "_" + VERSION, PATH)

learning_rate = 3e-3
num_epochs = 250 if not SMOKE_TEST else 1

n_data = 3000
dtype = torch.float
length_of_sequences = 10 + 1


if TYPE_OF_DATA == "ar5":
    y = return_arma_data(n_data)
elif TYPE_OF_DATA == "sin":
    y = np.sin(0.2 * np.linspace(0, n_data, n_data))
else:
    raise ValueError("invalid type of data")


sigma = 0.1
data = y + np.random.normal(0, sigma, n_data)
data /= data.max()

if SHOW_FIGURES:

    x = range(n_data)

    plt.plot(x, data, label="Data")
    plt.xlabel("time")
    plt.ylabel("y (value to forecast)")
    plt.fill_between(x, data + 2*sigma*np.ones(n_data), data - 2*sigma*np.ones(n_data), alpha=0.5)
    plt.legend()
    plt.show()

number_of_sequences = n_data - length_of_sequences + 1

sequences = list()
for sequence_index in range(number_of_sequences):
    sequence = data[sequence_index:sequence_index + length_of_sequences]
    sequences.append(sequence)

sequences = np.array(sequences)

# make training and test sets in torch
all_data = sequences[:, np.newaxis, :]
all_data = np.swapaxes(all_data, axis1=2, axis2=0)
all_data = np.swapaxes(all_data, axis1=2, axis2=1)

#print(training_data.shape)
training_data_labels = all_data[length_of_sequences-1, :, 0]

number_of_train_data = floor(0.67*n_data)

data_train = all_data[:, :number_of_train_data, :]

if SHOW_FIGURES:
    plt.plot(all_data[length_of_sequences-2, :, 0])
    plt.plot(training_data_labels)
    plt.legend()
    plt.show()

print("mean difference: ", np.mean(np.abs(all_data[length_of_sequences-2, :, 0] - training_data_labels)))


model = LSTM(lstm_params)


loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if TRAIN_LSTM:
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

    load_checkpoint(model, optimizer, PATH, NAME + "_" + VERSION)


if SAVE_LSTM:
    save_checkpoint(model, optimizer, PATH, NAME + "_" + VERSION)
    lstm_params.save(VERSION, PATH)

y_pred, _ = make_predictions(data_loader, model, all_data, lstm_params.batch_size)
features, y_true = extract_features(data_loader, model, all_data, lstm_params.batch_size)

print(np.corrcoef(features.T))
print("mean difference: ", np.mean(np.abs(y_pred[number_of_train_data:] - y_true[number_of_train_data:])))

if SHOW_FIGURES:
    plt.plot(y_pred, label="Preds")
    plt.plot(y_true, label="Data")
    plt.legend()
    plt.show()

    if TRAIN_LSTM:
        plt.plot(hist)
        plt.show()

X_train, X_test, y_train, y_test = features[:number_of_train_data], features[number_of_train_data:], \
                                   y_true[:number_of_train_data], y_true[number_of_train_data:]


priors_beta, _ = model.last_layers_weights

model_linear_mcmc = GaussianLinearModel_MCMC(X_train, y_train, priors_beta)
model_linear_mcmc.option = "hybrid"
model_linear_mcmc.sample()
model_linear_mcmc.show_trace()
predictions = model_linear_mcmc.make_predictions(X_test, y_test)

predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)

show_analysis(predictions.values, predictions.true_values, name="LSTM + " + model_linear_mcmc.option)

