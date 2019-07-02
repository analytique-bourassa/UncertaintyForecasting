from theano import shared
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

from probabilitic_predictions.probabilistic_predictions import ProbabilisticPredictions
from models.LSTM_BayesRegressor.GaussianLinearModel_abstract import GaussianLinearModel_abstract

class GaussianLinearModel_MCMC(GaussianLinearModel_abstract):

    def __init__(self, X_train, y_train, priors_beta=None):

        n_features = X_train.shape[1]

        self.X_data = X_train
        self.y_data = y_train
        self.mu_prior = y_train.mean()

        #Preprocess data for Modeling
        self.shA_X = shared(X_train)
        self.shA_y = shared(y_train)

        if priors_beta is None:
            priors_beta = 1.

        #Generate Model
        self.linear_model = pm.Model()

        with self.linear_model:
            # Priors for unknown model parameters
            alpha = pm.Normal("alpha", mu=self.mu_prior, sd=2)
            betas = pm.Normal("betas", mu=priors_beta, sd=2, shape=n_features)
            sigma = pm.HalfNormal("sigma", sd=10)  # you could also try with a HalfCauchy that has longer/fatter tails
            mu = alpha + pm.math.dot(betas, self.shA_X.T)
            self.likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=self.shA_y)

    def sample(self):

        SMOKE_TEST = False

        number_of_samples = 1000 if not SMOKE_TEST else 1
        number_of_tuning_step = 2000 if not SMOKE_TEST else 1

        with self.linear_model:
            step = pm.NUTS()
            self.trace = pm.sample(number_of_samples, step, tune=number_of_tuning_step)

    def show_trace(self):

        pm.traceplot(self.trace)
        plt.show()

    def make_predictions(self, X_test, y_test):

        number_of_samples = 1000

        predictions = ProbabilisticPredictions()
        predictions.number_of_predictions = X_test.shape[0]
        predictions.number_of_samples = number_of_samples
        predictions.initialize_to_zeros()

        # Prediction

        self.shA_X.set_value(X_test)

        zeros = np.zeros(X_test.shape[0], dtype=np.float32)
        self.shA_y.set_value(zeros)

        ppc = pm.sample_ppc(self.trace,
                            model=self.linear_model,
                            samples=number_of_samples)

        for idx in range(y_test.shape[0]):
            predictions.values[idx] = list(ppc.items())[0][1].T[idx]
            predictions.true_values[idx] = y_test[idx]

        return predictions

