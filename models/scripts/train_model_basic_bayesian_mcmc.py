import torch
import matplotlib.pyplot as plt
from models.LSTM_MCMC.LSTM import LSTM
from math import floor
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

def data_loader(data, batch_size, random=True):

    number_of_batches = int(floor(data.shape[1] / batch_size))
    if random:
        indexes_of_batch = np.random.choice(range(number_of_batches),
                                            number_of_batches, replace=False)
    else:
        indexes_of_batch = range(number_of_batches)

    for index_batch in indexes_of_batch:
        batch_first_index = index_batch * batch_size
        batch_last_index = (index_batch + 1) * batch_size

        X_numpy = data[:, batch_first_index:batch_last_index, 0]
        X_torch = torch.from_numpy(X_numpy).type(torch.Tensor)

        y_numpy = data[-1, batch_first_index:batch_last_index, 0]
        y_torch = torch.from_numpy(y_numpy).type(torch.Tensor)

        yield X_torch, y_torch

def make_forward_pass(data_loader, model, loss_fn):

    losses = None
    N_data = 0

    for X_train, y_train in data_loader(training_data, batch_size):

        N_data += batch_size
        y_pred = model(X_train)

        if losses is None:
            losses = loss_fn(y_pred, y_train)
        else:
            losses += loss_fn(y_pred, y_train)

    return losses, N_data

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    # Forward pass

    losses, N_data = make_forward_pass(data_loader, model, loss_fn)
    if t % 10 == 0:
        print("Epoch ", t, "MSE: ", losses.item())

    hist[t] = losses.item()

    optimiser.zero_grad()
    losses.backward()
    optimiser.step()

#####################
# Plot preds and performance
#####################


def make_predictions(data_loader, model):
    """
       NOTE: the feature index for the noise is based on the keeped indexes!!
    """
    model.eval()

    y_pred_all = None
    y_test_all = None

    for X_train, y_train in data_loader(training_data, batch_size, random=False):

        y_pred = model(X_train)


        if y_pred_all is None:
            y_pred_all = y_pred.detach().numpy()
            y_test_all = y_train.detach().numpy()

        y_pred_all = np.concatenate([y_pred_all, y_pred.detach().numpy()])
        y_test_all = np.concatenate([y_test_all, y_train.detach().numpy()])

    model.train()

    return y_pred_all, y_test_all


def extract_features(data_loader, model):
    """
       NOTE: the feature index for the noise is based on the keeped indexes!!
    """
    model.eval()

    y_pred_all = None
    y_test_all = None

    for X_train, y_train in data_loader(training_data, batch_size, random=False):

        y_pred = model.return_last_layer(X_train)


        if y_pred_all is None:
            y_pred_all = y_pred.detach().numpy()
            y_test_all = y_train.detach().numpy()

        y_pred_all = np.concatenate([y_pred_all, y_pred.detach().numpy()])
        y_test_all = np.concatenate([y_test_all, y_train.detach().numpy()])

    model.train()

    return y_pred_all, y_test_all



y_pred, y_true = make_predictions(data_loader, model)
features, y_true = extract_features(data_loader, model)

print(np.corrcoef(features.T))
if SHOW_FIGURES:
    plt.plot(y_pred, label="Preds")
    plt.plot(y_true, label="Data")
    plt.legend()
    plt.show()

    plt.plot(hist)
    plt.show()

import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import matplotlib.pyplot as plt



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
shA_X = shared(X)

#Generate Model
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
