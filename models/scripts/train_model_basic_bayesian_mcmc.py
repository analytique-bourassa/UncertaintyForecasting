import torch
import matplotlib.pyplot as plt
from models.LSTM_MCMC.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np
from models.LSTM_MCMC.gaussian_model_mcmc import GaussianLinearModel_MCMC
from models.calibration.analysis import show_analysis
#####################
# Set parameters
#####################
SHOW_FIGURES = False
# Data params
noise_var = 0
num_datapoints = 300
test_size = 0.2
num_train = int((1 - test_size) * num_datapoints)

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
num_epochs = 250
dropout = 0.4

n_data = 300
dtype = torch.float
length_of_sequences = 10


#####################
# Generate data_handling
#####################

data = np.sin(0.2*np.linspace(0, 200, n_data)) + np.random.normal(0, 0.1, n_data)

number_of_sequences = n_data - length_of_sequences+ 1

sequences = list()
for sequence_index in range(number_of_sequences):
    sequence = data[sequence_index:sequence_index + length_of_sequences]
    sequences.append(sequence)

sequences = np.array(sequences)



# make training and test sets in torch
training_data = sequences[:, np.newaxis, :]
training_data = np.swapaxes(training_data, axis1=2, axis2=0)
training_data = np.swapaxes(training_data, axis1=2, axis2=1)



training_data_labels = training_data[length_of_sequences-1, : , 0]

if SHOW_FIGURES:
    plt.plot(training_data[length_of_sequences-1, :, 0])
    plt.plot(training_data_labels)
    plt.legend()
    plt.show()

#####################
# Build model
#####################
batch_size = 10

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

    losses, N_data = make_forward_pass(data_loader, model, loss_fn, training_data, batch_size)
    if t % 10 == 0:
        print("Epoch ", t, "MSE: ", losses.item())

    hist[t] = losses.item()

    optimiser.zero_grad()
    losses.backward()
    optimiser.step()

#####################
# Plot preds and performance
#####################


y_pred, y_true = make_predictions(data_loader, model, training_data, batch_size)
features, y_true = extract_features(data_loader, model, training_data, batch_size)

print(np.corrcoef(features.T))
if SHOW_FIGURES:
    plt.plot(y_pred, label="Preds")
    plt.plot(y_true, label="Data")
    plt.legend()
    plt.show()

    plt.plot(hist)
    plt.show()


np.random.seed(9)

X = features
y_ = y_pred

X_train, X_test, y_train, y_test = X[:275], X[275:], y_[:275], y_[275:]

print(X.shape, y_.shape, X_train.shape, X_test.shape)



model_linear_mcmc = GaussianLinearModel_MCMC(X_train, y_train)
model_linear_mcmc.sample()
model_linear_mcmc.show_trace()
predictions = model_linear_mcmc.make_predictions(X_test, y_test)

predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)

show_analysis(predictions.values, predictions.true_values, name="LSTM + MCMC")