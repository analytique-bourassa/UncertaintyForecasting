import torch
import matplotlib.pyplot as plt
from models.LSTM_MCMC.LSTM import LSTM
from math import floor
import numpy as np
from data_generation.data_generator import return_arma_data, return_sinus_data

#####################
# Set parameters
#####################

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

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

# for CI testing
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.3.3')
pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)


DATA_URL = "https://d2fefpcigoriu7.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = data[data["cont_africa"] == 1]
non_african_nations = data[data["cont_africa"] == 0]
sns.scatterplot(non_african_nations["rugged"],
            np.log(non_african_nations["rgdppc_2000"]),
            ax=ax[0])
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
sns.scatterplot(african_nations["rugged"],
            np.log(african_nations["rgdppc_2000"]),
            ax=ax[1])
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")

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
num_iterations = 1000 if not smoke_test else 2
data = torch.tensor(df.values, dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]

x_data = torch.tensor(features, dtype=torch.float)
y_data = torch.tensor(y_true, dtype=torch.float)

def main():
    x_data = torch.tensor(features, dtype=torch.float)#data[:, :-1]
    y_data = torch.tensor(y_true, dtype=torch.float) #data[:, -1]

    for j in range(num_iterations):
        # run the model forward on the data
        y_pred = regression_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in regression_model.named_parameters():
        print(name, param.data.numpy())

main()

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
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

train()

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))

get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()


def summary(traces, sites):

    marginal = get_marginal(traces, sites)
    site_stats = {}

    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = pd.DataFrame(marginal[:, i]).transpose()
        describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
        site_stats[site_name] = marginal_site.apply(describe, axis=1) \
            [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]

    return site_stats

def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))

posterior = svi.run(x_data, y_data)


# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=1000)

post_pred = trace_pred.run(x_data, None)
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
mu = post_summary["prediction"]
y = post_summary["obs"]
predictions = pd.DataFrame({
    "cont_africa": x_data[:, 0],
    "rugged": x_data[:, 1],
    "mu_mean": mu["mean"],
    "mu_perc_5": mu["5%"],
    "mu_perc_95": mu["95%"],
    "y_mean": y["mean"],
    "y_perc_5": y["5%"],
    "y_perc_95": y["95%"],
    "true_gdp": y_data,
})
plt.show()


plt.plot(y["mean"])
plt.show()

x = range(y_true.shape[0])
plt.plot(x, y["mean"], label="Preds")
plt.plot(x, y_true, label="Data")
plt.fill_between(x, y["5%"], y["95%"], alpha=0.5)
plt.legend()
plt.show()


# we need to prepend `module$$$` to all parameters of nn.Modules since
# that is how they are stored in the ParamStore
weight = get_marginal(posterior, ['module$$$linear.weight']).squeeze(1).squeeze(1)
factor = get_marginal(posterior, ['module$$$factor'])
gamma_within_africa = weight[:, 1] + factor.squeeze(1)
gamma_outside_africa = weight[:, 1]

fig = plt.figure(figsize=(10, 6))
sns.distplot(gamma_within_africa, kde_kws={"label": "African nations"},)
sns.distplot(gamma_outside_africa, kde_kws={"label": "Non-African nations"})
fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness", fontsize=16)
plt.show()