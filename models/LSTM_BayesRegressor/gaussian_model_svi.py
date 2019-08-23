import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.optim import Adam
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.contrib.autoguide import AutoDiagonalNormal

import torch
import torch.nn as nn

from probabilitic_predictions.probabilistic_predictions_regression import ProbabilisticPredictionsRegression
from models.ProbabilisticModelWrapperAbstract import GaussianLinearModel_abstract

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


class GaussianLinearModel_SVI(GaussianLinearModel_abstract):

    def __init__(self, X_train, y_train, priors_beta=None):

        self.number_of_features = X_train.shape[1]
        self.regression_model = RegressionModel(self.number_of_features)

        self.X_train = torch.tensor(X_train, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.float)

        guide = AutoDiagonalNormal(self.model)
        optim = Adam({"lr": 0.03})

        self.svi = SVI(self.model, guide, optim, loss=Trace_ELBO(), num_samples=1000)

    def sample(self):
        num_iterations = 1000
        pyro.clear_param_store()
        for j in range(num_iterations):
            # calculate the loss and take a gradient step
            loss = self.svi.step(self.X_train, self.y_train)
            if j % 100 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(self.X_train)))

        self.posterior = self.svi.run(self.X_train, self.y_train)

        self.trace_pred = TracePredictive(self.wrapped_model,
                                          self.posterior,
                                     num_samples=1000)

    def show_trace(self):
        raise NotImplementedError

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

        w_prior = Normal(torch.zeros(1, self.number_of_features),
                         torch.ones(1, self.number_of_features)).to_event(1)

        b_prior = Normal(torch.tensor([[8.]]), torch.tensor([[1000.]])).to_event(1)
        f_prior = Normal(0., 1.)

        priors = {'linear.weight': w_prior,
                  'linear.bias': b_prior,
                  'factor': f_prior}

        scale = pyro.sample("sigma", Uniform(0., 10.))

        lifted_module = pyro.random_module("module", self.regression_model, priors)
        lifted_reg_model = lifted_module()

        with pyro.plate("map", len(x_data)):
            prediction_mean = lifted_reg_model(x_data).squeeze(-1)
            # condition on the observed data
            pyro.sample("obs",
                        Normal(prediction_mean, scale),
                        obs=y_data)

            return prediction_mean
