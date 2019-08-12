import matplotlib.pyplot as plt
from models.LSTM_BayesRegressor.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np
from models.calibration.analysis import show_analysis
from models.disk_reader_and_writer import save_checkpoint
from models.lstm_params import LSTM_parameters
from models.LSTM_BayesRegressor.gaussian_model_svi import GaussianLinearModel_SVI

from data_generation.data_generator import return_arma_data

#####################
# Set parameters
#####################
PATH = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

VERSION = "v0.1.0"
SHOW_FIGURES = True
SMOKE_TEST = False
TRAIN_LSTM = True
SAVE_LSTM = True

# Network params
lstm_params = LSTM_parameters()
learning_rate = 3e-3
num_epochs = 250 if not SMOKE_TEST else 1

n_data = 3000
dtype = torch.float
length_of_sequences = 10 + 1



#####################
# Generate data_handling
#####################

y = return_arma_data(n_data)
plt.plot(y)
plt.show()

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
#####################
# Build model
#####################
batch_size = 20

model = LSTM(lstm_params)

loss_fn = torch.nn.MSELoss(size_average=False)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)


for t in range(num_epochs):

    model.hidden = model.init_hidden()

    losses, N_data = make_forward_pass(data_loader_sequences, model, loss_fn, data_train, batch_size)
    if t % 10 == 0:
        print("Epoch ", t, "MSE: ", losses.item())

    hist[t] = losses.item()

    optimiser.zero_grad()
    losses.backward()
    optimiser.step()

#####################
# Plot preds and performance
#####################

if SAVE_LSTM:
    save_checkpoint(model, optimiser, PATH, "feature_extractor_" + VERSION)
    lstm_params.save(VERSION, PATH)

y_pred, _ = make_predictions(data_loader_sequences, model, all_data, batch_size)
features, y_true = extract_features(data_loader_sequences, model, all_data, batch_size)

print(np.corrcoef(features.T))
print("mean difference: ", np.mean(np.abs(y_pred[number_of_train_data:] - y_true[number_of_train_data:])))

if SHOW_FIGURES:
    plt.plot(y_pred, label="Preds")
    plt.plot(y_true, label="Data")
    plt.legend()
    plt.show()

    plt.plot(hist)
    plt.show()


np.random.seed(9)

X_train, X_test, y_train, y_test = features[:number_of_train_data], features[number_of_train_data:], \
                                   y_true[:number_of_train_data], y_true[number_of_train_data:]


model_linear_svi = GaussianLinearModel_SVI(X_train, y_train)
model_linear_svi.sample()
predictions = model_linear_svi.make_predictions(X_test, y_test)

predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)
show_analysis(predictions.values, predictions.true_values, name="LSTM + SVI")



