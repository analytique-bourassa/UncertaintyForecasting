import torch
import matplotlib.pyplot as plt
from models.LSTM_BayesRegressor.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np
from models.LSTM_BayesRegressor.bayesian_linear_regression import BayesianLinearModel
from models.calibration.analysis import show_analysis


#####################
# Set parameters
#####################
SHOW_FIGURES = False
SMOKE_TEST = False

# Network params
input_size = 20

per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
# size of hidden layers

h1 = 5
output_dim = 1
num_layers = 1
learning_rate = 3e-3
num_epochs = 250 if not SMOKE_TEST else 1
dropout = 0.4

n_data = 3000
dtype = torch.float
length_of_sequences = 10 + 1



#####################
# Generate data_handling
#####################
sigma = 0.1
data = np.sin(0.2*np.linspace(0, n_data, n_data)) + np.random.normal(0, sigma, n_data)
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

model = LSTM(1,
             h1,
             batch_size=batch_size,
             output_dim=output_dim,
             num_layers=num_layers,
             dropout=dropout)

loss_fn = torch.nn.MSELoss(size_average=False)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)


for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    # Forward pass

    losses, N_data = make_forward_pass(data_loader, model, loss_fn, data_train, batch_size)
    if t % 10 == 0:
        print("Epoch ", t, "MSE: ", losses.item())

    hist[t] = losses.item()

    optimiser.zero_grad()
    losses.backward()
    optimiser.step()

#####################
# Plot preds and performance
#####################


y_pred, _ = make_predictions(data_loader, model, all_data, batch_size)
features, y_true = extract_features(data_loader, model, all_data, batch_size)

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


priors_beta, _ = model.last_layers_weights

model_linear_mcmc = BayesianLinearModel(X_train, y_train, priors_beta)
model_linear_mcmc.sample()
model_linear_mcmc.show_trace()
predictions = model_linear_mcmc.make_predictions(X_test, y_test)

predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)

show_analysis(predictions.values, predictions.true_values, name="LSTM + MCMC")

