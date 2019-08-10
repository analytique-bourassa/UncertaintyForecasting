import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.optim import Adam
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer.mcmc import MCMC, NUTS

import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from probabilitic_predictions.probabilistic_predictions_regression import ProbabilisticPredictionsRegression
from models.LSTM_BayesRegressor.GaussianLinearModel_abstract import GaussianLinearModel_abstract

get_marginal = lambda traces, sites: EmpiricalMarginal(traces, sites)._get_samples_and_weights()[
        0].detach().cpu().numpy()

class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.linear(x) + (self.factor * x[:, 0] * x[:, 1]).unsqueeze(1)


class GaussianLinearModel_MCMC_pyro(GaussianLinearModel_abstract):

    def __init__(self, X_train, y_train, priors_beta=None):

        self.number_of_features = X_train.shape[1]
        self.regression_model = RegressionModel(self.number_of_features)

        self.X_train = torch.tensor(X_train, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.float)

    def sample(self):

        nuts_kernel = NUTS(self.model, adapt_step_size=False, step_size=0.1)
        self.posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200) \
            .run(self.X_train, self.y_train)


        #    ax.set_title(sites[i])
       # handles, labels = ax.get_legend_handles_labels()
       # fig.legend(handles, labels, loc='upper right')

        self.trace_pred = TracePredictive(self.wrapped_model,
                                          self.posterior,
                                     num_samples=1000)

    def show_trace(self):

        return 0
        sites = ["sigma", "betas", "alpha"]
        hmc_empirical = EmpiricalMarginal(self.posterior, sites=sites)
        #._get_samples_and_weights()[0].numpy()
        # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        # fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
        # for i, ax in enumerate(axs.reshape(-1)):
        # sns.distplot(svi_diagnorm_empirical[:, i], ax=ax, label="SVI (DiagNormal)")
        sns.distplot(hmc_empirical[:, 0])
        plt.show()

    def make_predictions(self, X_test, y_test):

        number_of_samples = 1000

        X_test = torch.tensor(X_test, dtype=torch.float)

        post_pred = self.trace_pred.run(X_test, None)
        mu, y = self.get_results(post_pred, sites=['prediction', 'obs'])

        predictions = ProbabilisticPredictionsRegression()
        predictions.number_of_predictions = X_test.shape[0]
        predictions.number_of_samples = number_of_samples
        predictions.initialize_to_zeros()

        predictions.values = y
        predictions.true_values = y_test

        return predictions


    def get_results(self, traces, sites):

        marginal = get_marginal(traces, sites)

        return marginal[:, 0, :].transpose(), marginal[:, 1, :].transpose()

    def wrapped_model(self, x_data, y_data):
        pyro.sample("prediction", Delta(self.model(x_data, y_data)))

    def model(self, x_data, y_data):

        alpha = pyro.sample("alpha", Normal(0, 1))
        betas = pyro.sample("betas", Normal(torch.zeros(self.number_of_features),
                         torch.ones(self.number_of_features)).to_event(1))
        scale = pyro.sample("sigma", Uniform(0., 10.))

        for i in pyro.plate("observations", len(x_data)):
            prediction_mean = alpha + torch.dot(betas, x_data[i])
            pyro.sample("obs",
                        Normal(prediction_mean, scale),
                        obs=y_data[i])

            return prediction_mean
