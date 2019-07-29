import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from pymc3.distributions import Interpolated
from theano import shared

from probabilitic_predictions.probabilistic_predictions import ProbabilisticPredictions
from models.LSTM_BayesRegressor.GaussianLinearModel_abstract import GaussianLinearModel_abstract
from models.LSTM_BayesRegressor.bayesian_linear_regression_parameters import BayesianLinearRegressionParameters

def from_posterior(param, samples):

    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)

class BayesianLinearModel(GaussianLinearModel_abstract):

    POSSIBLE_OPTION_FOR_POSTERIOR_CALCULATION = ["NUTS",
                                                 "ADVI-Mean-Field",
                                                 "ADVI-full-rank",
                                                 "Hybrid"]

    KEY_INDEX_FOR_NUMBER_OF_DATA = 0

    def __init__(self, X_train, y_train, priors_beta=None, SMOKE_TEST=False):

        self.option = "ADVI-Mean-Field"

        self.X_data = X_train
        self.y_data = y_train

        self.mu_prior = y_train.mean()
        self.n_features = X_train.shape[1]

        self.shared_X = shared(X_train)
        self.shared_y = shared(y_train)

        self.params = BayesianLinearRegressionParameters(SMOKE_TEST)

        if priors_beta is None:
            priors_beta = 1.

        self.linear_model = pm.Model()

        with self.linear_model:
            alpha = pm.Normal("alpha", mu=self.mu_prior, sd=2)
            betas = pm.Normal("betas", mu=priors_beta, sd=2, shape=self.n_features)
            sigma = pm.HalfNormal("sigma", sd=10)
            mu = alpha + pm.math.dot(betas, self.shared_X.T)
            self.likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=self.shared_y)

    def sample(self):

        if self.option == "NUTS":
            with self.linear_model:

                step = pm.NUTS()
                self.trace = pm.sample(self.params.number_of_samples_for_posterior,
                                       step,
                                       tune=self.params.number_of_tuning_steps)

        elif self.option == "ADVI-Mean-Field":
            with self.linear_model:

                self.advi_fit = pm.fit(method=pm.ADVI(),
                                       n=self.params.number_of_iterations)

                self.trace = self.advi_fit.sample(self.params.number_of_samples_for_posterior)

        elif self.option == "ADVI-full-rank":
            with self.linear_model:
                self.advi_fit = pm.fit(method='fullrank_advi',
                                       n=self.params.number_of_iterations)

                self.trace = self.advi_fit.sample(self.params.number_of_samples_for_posterior)

        elif self.option == "Hybrid":
            with self.linear_model:

                self.advi_fit = pm.fit(method=pm.ADVI(),
                                       n=self.params.number_of_iterations)

                self.trace = self.advi_fit.sample(self.params.number_of_samples_for_posterior)

                trace_alpha = self.trace.get_values('alpha')
                trace_betas = self.trace.get_values('betas')
                trace_sigma = self.trace.get_values('sigma')

            self.linear_model_2 = pm.Model()

            with self.linear_model_2:

                alpha_mu_prior = trace_alpha.mean()
                alpha_sd_prior = trace_alpha.std()

                alpha = pm.Normal("theta_0",
                                  mu=alpha_mu_prior,
                                  sd=alpha_sd_prior)

                betas = pm.Normal("thetas",
                                  mu=trace_betas.mean(axis=0),
                                  sd=trace_betas.std(axis=0),
                                  shape=self.n_features)

                sigma = from_posterior("sigma", trace_sigma)

                mu = alpha + pm.math.dot(betas, self.shared_X.T)
                self.likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=self.shared_y)

                step = pm.NUTS()
                self.trace = pm.sample(self.params.number_of_samples_for_posterior,
                                       step,
                                       tune=self.params.number_of_tuning_steps)

        else:
            raise ValueError("self.option not valid")

    def show_trace(self):

        pm.traceplot(self.trace)
        plt.show()

    def make_predictions(self, X_test, y_test):

        predictions = ProbabilisticPredictions()
        predictions.number_of_predictions = X_test.shape[0]
        predictions.number_of_samples = self.params.number_of_samples_for_predictions
        predictions.initialize_to_zeros()

        # Prediction

        self.shared_X.set_value(X_test)

        zeros = np.zeros(X_test.shape[0], dtype=np.float32)
        self.shared_y.set_value(zeros)

        if self.option == "Hybrid":
            ppc = pm.sample_ppc(self.trace,
                                model=self.linear_model_2,
                                samples=self.params.number_of_samples_for_predictions)

        else:
            ppc = pm.sample_ppc(self.trace,
                                model=self.linear_model,
                                samples=self.params.number_of_samples_for_predictions)

        for idx in range(y_test.shape[self.KEY_INDEX_FOR_NUMBER_OF_DATA]):
            predictions.values[idx] = list(ppc.items())[0][1].T[idx]
            predictions.true_values[idx] = y_test[idx]

        return predictions
