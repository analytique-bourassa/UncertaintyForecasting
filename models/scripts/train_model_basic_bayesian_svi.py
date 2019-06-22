import torch
import matplotlib.pyplot as plt
from models.LSTM_MCMC.LSTM import LSTM
from models.model_data_feeder import *
import numpy as np
from models.LSTM_MCMC.gaussian_model_mcmc import GaussianLinearModel_MCMC
from models.calibration.analysis import show_analysis
from models.disk_reader_and_writer import save_checkpoint

#####################
# Set parameters
#####################
SHOW_FIGURES = False
SMOKE_TEST = False
TRAIN_LSTM = True
SAVE_LSTM = True

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

if SAVE_LSTM:

    path = "/home/louis/Documents/ConsultationSimpliphAI/" \
           "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_MCMC/.models/"
    save_checkpoint(model, optimiser, path, "feature_extractor_v0.0.2")

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



import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.optim import Adam
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive

import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.linear(x) + (self.factor * x[:, 0] * x[:, 1]).unsqueeze(1)

p = h1 #2  # number of features
regression_model = RegressionModel(p)

loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(regression_model.parameters(), lr=0.05)
num_iterations = 1000 if not SMOKE_TEST else 2


X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)


loc = torch.zeros(1, 1)
scale = torch.ones(1, 1)
# define a unit normal prior
prior = Normal(loc, scale)
# overload the parameters in the regression module with samples from the prior
lifted_module = pyro.random_module("regression_module", regression_model, prior)
# sample a nn from the prior
sampled_reg_model = lifted_module()

def model(x_data, y_data):
    # weight and bias priors
    w_prior = Normal(torch.zeros(1, p), torch.ones(1, p)).to_event(1)
    b_prior = Normal(torch.tensor([[8.]]), torch.tensor([[1000.]])).to_event(1)
    f_prior = Normal(0., 1.)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior, 'factor': f_prior}
    scale = pyro.sample("sigma", Uniform(0., 10.))
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.plate("map", len(x_data)):
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data)
        return prediction_mean

from pyro.contrib.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model)

optim = Adam({"lr": 0.03})
svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)

def train():
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(X_train, y_train)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

train()

get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()


def get_results(traces, sites):

    marginal = get_marginal(traces, sites)

    return marginal[:, 0, :].transpose(), marginal[:, 1, :].transpose()



def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))


posterior = svi.run(X_train, y_train)

# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=1000)

X_test = torch.tensor(X_test, dtype=torch.float)
post_pred = trace_pred.run(X_test, None)
mu, y = get_results(post_pred, sites= ['prediction', 'obs'])


number_of_samples = 1000

from probabilitic_predictions.probabilistic_predictions import ProbabilisticPredictions

predictions = ProbabilisticPredictions()
predictions.number_of_predictions = X_test.shape[0]
predictions.number_of_samples = number_of_samples
predictions.initialize_to_zeros()

predictions.values = y
predictions.true_values = y_test

predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)

show_analysis(predictions.values, predictions.true_values, name="LSTM + SVI")


"""
priors_beta, _ = model.last_layers_weights

model_linear_mcmc = GaussianLinearModel_MCMC(X_train, y_train, priors_beta)
model_linear_mcmc.sample()
model_linear_mcmc.show_trace()
predictions = model_linear_mcmc.make_predictions(X_test, y_test)

predictions.show_predictions_with_confidence_interval(confidence_interval=0.95)

show_analysis(predictions.values, predictions.true_values, name="LSTM + MCMC")

"""


