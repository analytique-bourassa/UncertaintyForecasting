import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

from theano import shared

from models.LSTM_BayesRegressor.bayesian_linear_regression.bayesian_linear_regression_parameters import BayesianLinearRegressionParameters
from models.ProbabilisticModelWrapperAbstract import ProbabilisticModelWrapperAbstract
from probabilitic_predictions.probabilistic_predictions_classification import ProbabilisticPredictionsClassification
from utils.timers import TimerContext
from utils.time_profiler_logging import TimeProfilerLogger

logger_time = TimeProfilerLogger.getInstance()

class BayesianSoftmaxClassificationWithTemperatures(ProbabilisticModelWrapperAbstract):

    KEY_INDEX_FOR_NUMBER_OF_DATA = 0

    def __init__(self, number_of_features, number_of_classes, X_train, y_train):

        super().__init__()

        self.X_data = X_train
        self.y_data = y_train

        self.n_features = X_train.shape[1]

        self.shared_X = shared(X_train)
        self.shared_y = shared(y_train)

        self.params = BayesianLinearRegressionParameters()
        self.number_of_classes = number_of_classes
        self.number_of_features = number_of_features

        with pm.Model() as classification_model:

            theta_0 = pm.Normal('theta_0', mu=0, sd=1.0, shape=self.number_of_classes)

            thetas = pm.Normal('thetas', mu=0, sd=1.0, shape=(self.number_of_features,
                                                             self.number_of_classes))

            temperatures = pm.HalfNormal('temperatures',
                                         sd=10.0,
                                         shape=self.number_of_classes)

            mu = theta_0 + pm.math.dot(self.shared_X, thetas)
            mu_with_temperatures = mu*temperatures

            probability_of_class = pm.math.exp(mu_with_temperatures) / \
                                   pm.math.sum(pm.math.exp(mu_with_temperatures), axis=0)

            self.likelihood = pm.Categorical('category',
                                             p=probability_of_class,
                                             observed=self.shared_y)

            self.classification_model = classification_model


    def sample(self):
        with TimerContext(name="sampling",
                          show_time_when_exit=True,
                          logger=logger_time):

            with self.classification_model:

                self.advi_fit = pm.fit(method=pm.ADVI(),
                                       n=self.params.number_of_iterations,
                                       progressbar=self.show_progress)

                self.trace = self.advi_fit.sample(self.params.number_of_samples_for_posterior)

                trace_theta_0 = self.trace.get_values('theta_0')
                trace_thetas = self.trace.get_values('thetas')
                trace_temperatures = self.trace.get_values('temperatures')

            self.model = pm.Model()

            with self.model:


                alpha = pm.Normal("theta_0",
                                  mu=trace_theta_0.mean(),
                                  sd=trace_thetas.std(),
                                  shape=self.number_of_classes)

                betas = pm.Normal("thetas",
                                  mu=trace_thetas.mean(axis=0),
                                  sd=trace_thetas.std(axis=0),
                                  shape=(self.number_of_features,
                                         self.number_of_classes))

                mu = alpha + pm.math.dot(self.shared_X, betas)

                temperatures = pm.HalfNormal('temperatures',
                                         sd=trace_temperatures.std(),
                                         shape=self.number_of_classes)

                mu_with_temperatures = mu*temperatures

                probability_of_class = pm.math.exp(mu_with_temperatures) / \
                                       pm.math.sum(pm.math.exp(mu_with_temperatures), axis=0)


                self.likelihood = pm.Categorical('category',
                                                 p=probability_of_class,
                                                 observed=self.shared_y)

                self.trace = pm.sample(self.params.number_of_samples_for_posterior,
                                       tune=self.params.number_of_tuning_steps,
                                       progressbar=self.show_progress)

    def show_trace(self):

        pm.traceplot(self.trace)
        plt.show()

    def make_predictions(self, X_test, y_test):

        self.shared_X.set_value(X_test)

        zeros = np.zeros(X_test.shape[0], dtype=np.int64)
        self.shared_y.set_value(zeros)

        ppc = pm.sample_ppc(self.trace,
                            model=self.model,
                            samples=self.params.number_of_samples_for_predictions,
                            progressbar=self.show_progress)

        predictions = ProbabilisticPredictionsClassification(number_of_classes=self.number_of_classes)
        predictions.number_of_predictions = X_test.shape[0]
        predictions.number_of_samples = self.params.number_of_samples_for_predictions
        predictions.initialize_to_zeros()

        index_dummy_axis = 0
        index_for_values = 1

        samples = list(ppc.items())[index_dummy_axis][index_for_values].T
        predictions.values = samples
        predictions.true_values = y_test

        return predictions


