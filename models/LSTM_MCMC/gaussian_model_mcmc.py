from theano import shared
import pymc3 as pm
import matplotlib.pyplot as plt

from probabilitic_predictions.probabilistic_predictions import ProbabilisticPredictions

class GaussianLinearModel_MCMC():

    def __init__(self, X_train, y_train):

        n_features = X_train.shape[1]
        self.X_data = X_train
        self.y_data = y_train
        self.mu_prior = y_train.mean()
        #Preprocess data for Modeling
        self.shA_X = shared(X_train)


        #Generate Model
        self.linear_model = pm.Model()

        with self.linear_model:
            # Priors for unknown model parameters
            alpha = pm.Normal("alpha", mu=self.mu_prior, sd=2)
            betas = pm.Normal("betas", mu=1, sd=2, shape=n_features)
            sigma = pm.HalfNormal("sigma", sd=10)  # you could also try with a HalfCauchy that has longer/fatter tails
            mu = alpha + pm.math.dot(betas, self.shA_X.T)
            self.likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=y_train)

    def sample(self):

        with self.linear_model:
            step = pm.NUTS()
            self.trace = pm.sample(1000, step, tune=2000)

    def show_trace(self):

        #Traceplot
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
        ppc = pm.sample_ppc(self.trace,
                            model=self.linear_model,
                            samples=number_of_samples)


        for idx in range(y_test.shape[0]):
            predictions.values[idx] = list(ppc.items())[0][1].T[idx]
            predictions.true_values[idx] = y_test[idx]

        return predictions

#Looks like I need to transpose it to get `X_te` samples on rows and posterior distribution samples on cols
