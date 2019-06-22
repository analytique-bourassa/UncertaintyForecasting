import torch
import matplotlib.pyplot as plt
from models.LSTM_MCMC.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np
from data_generation.data_generator import return_arma_data, return_sinus_data

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

data = np.sin(0.2*np.linspace(0, 200, n_data))
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





import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy import stats, optimize
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from theano import shared

np.random.seed(9)


#Load the Data
diabetes_data = load_diabetes()
X = features
y_ = y_pred
#Split Data
X_tr, X_te, y_tr, y_te = X[:275], X[275:], y_[:275], y_[275:]

#Shapes
print(X.shape, y_.shape, X_tr.shape, X_te.shape)
#((442, 10), (442,), (331, 10), (111, 10))

#Preprocess data for Modeling
#shA_X = shared(X)

from models.LSTM_MCMC.gaussian_model_mcmc import GaussianLinearModel_MCMC

model_linear_mcmc = GaussianLinearModel_MCMC(X, y_)
model_linear_mcmc.sample()
model_linear_mcmc.show_trace()
predictions = model_linear_mcmc.make_predictions()

predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)
#Generate Model
"""
linear_model = pm.Model()

with linear_model:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=y_.mean(), sd=2)
    betas = pm.Normal("betas", mu=1, sd=2, shape=X.shape[1])
    sigma = pm.HalfNormal("sigma", sd=10)  # you could also try with a HalfCauchy that has longer/fatter tails
    mu = alpha + pm.math.dot(betas, X.T)
    likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=y_)
    step = pm.NUTS()
    trace = pm.sample(1000, step, tune=2000)

#Traceplot
pm.traceplot(trace)
plt.show()

# Prediction
#shA_X.set_value(X_te)
ppc = pm.sample_ppc(trace, model=linear_model, samples=1000)

#What's the shape of this?
list(ppc.items())[0][1].shape #(1000, 111) it looks like 1000 posterior samples for the 111 test samples (X_te) I gave it

#Looks like I need to transpose it to get `X_te` samples on rows and posterior distribution samples on cols

predicted_yi_list = list()
actual_yi_list = list()

for idx in range(y_.shape[0]):

    predicted_yi = list(ppc.items())[0][1].T[idx].mean()
    actual_yi = y_[idx]

    predicted_yi_list.append(predicted_yi)
    actual_yi_list.append(actual_yi)


plt.plot(predicted_yi_list, label="Preds")
plt.plot(actual_yi_list, label="Data")
plt.legend()
plt.show()
"""