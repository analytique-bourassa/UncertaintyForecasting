import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

from theano import shared

from uncertainty_forecasting.probabilitic_predictions.probabilistic_predictions_regression import ProbabilisticPredictionsRegression
from uncertainty_forecasting.models.ProbabilisticModelWrapperAbstract import ProbabilisticModelWrapperAbstract
from uncertainty_forecasting.models.regression.LSTM_BayesRegressor.bayesian_linear_regression.bayesian_linear_regression_parameters import BayesianLinearRegressionParameters
from uncertainty_forecasting.models.regression.LSTM_BayesRegressor.bayesian_linear_regression.bayesian_linear_regression_priors import BayesianLinearRegressionPriors
from uncertainty_forecasting.models.tools_pymc import from_posterior

from .sampling_method_enum import SamplingMethod


class BayesianLinearModel(ProbabilisticModelWrapperAbstract):
    """

    y ~ Normal(mu,sigma)
    mu = sum_i=1 theta_i*z_i + theta_0
    """

    POSSIBLE_OPTION_FOR_POSTERIOR_CALCULATION = [e.value for e in SamplingMethod]


    KEY_INDEX_FOR_NUMBER_OF_DATA = 0

    def __init__(self, X_train, y_train, priors_thetas=None, SMOKE_TEST=False):

        self.option = SamplingMethod.ADVI_MEAN_FIELD.value

        ################################
        self.X_data = X_train
        self.y_data = y_train

        self.n_features = X_train.shape[1]

        self.shared_X = shared(X_train)
        self.shared_y = shared(y_train)
        ################################

        self.params = BayesianLinearRegressionParameters()
        if SMOKE_TEST:
            self.params.set_for_smoke_test()

        self.priors = BayesianLinearRegressionPriors()


        self.linear_model = pm.Model()

        with self.linear_model:

            theta_0 = pm.Normal("theta_0",
                                mu=self.priors.mean_theta_0,
                                sd=self.priors.standard_deviation_theta_0
                                )

            thetas = pm.Normal("thetas",
                               mu=self.priors.mean_thetas,
                               sd=self.priors.standard_deviation_thetas,
                               shape=self.n_features)

            sigma = pm.HalfNormal("sigma",
                                  sd=self.priors.standard_deviation_sigma)

            mu = theta_0 + pm.math.dot(thetas, self.shared_X.T)

            self.likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=self.shared_y)

    def sample(self):

        if self.option == SamplingMethod.NUTS.value:
            with self.linear_model:

                step = pm.NUTS()
                self.trace = pm.sample(self.params.number_of_samples_for_posterior,
                                       step,
                                       tune=self.params.number_of_tuning_steps)

        elif self.option == SamplingMethod.ADVI_MEAN_FIELD.value:
            with self.linear_model:

                self.advi_fit = pm.fit(method=pm.ADVI(),
                                       n=self.params.number_of_iterations)

                self.trace = self.advi_fit.sample(self.params.number_of_samples_for_posterior)

        elif self.option == SamplingMethod.ADVI_FULL_RANK.value:
            with self.linear_model:
                self.advi_fit = pm.fit(method='fullrank_advi',
                                       n=self.params.number_of_iterations)

                self.trace = self.advi_fit.sample(self.params.number_of_samples_for_posterior)

        elif self.option == SamplingMethod.HYBRID.value:
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

        predictions = ProbabilisticPredictionsRegression()
        predictions.number_of_predictions = X_test.shape[0]
        predictions.number_of_samples = self.params.number_of_samples_for_predictions
        predictions.initialize_to_zeros()

        # Prediction

        self.shared_X.set_value(X_test)

        zeros = np.zeros(X_test.shape[0], dtype=np.float32)
        self.shared_y.set_value(zeros)

        if self.option == SamplingMethod.HYBRID.value:
            ppc = pm.sample_ppc(self.trace,
                                model=self.linear_model_2,
                                samples=self.params.number_of_samples_for_predictions)

        else:
            ppc = pm.sample_ppc(self.trace,
                                model=self.linear_model,
                                samples=self.params.number_of_samples_for_predictions)

        predictions.values = list(ppc.items())[0][1].T
        predictions.true_values = y_test

        return predictions

