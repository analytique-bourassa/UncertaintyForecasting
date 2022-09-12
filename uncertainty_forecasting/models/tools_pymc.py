import numpy as np

from scipy import stats
from pymc3.distributions import Interpolated
from theano import shared

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


class Dataset():

    def __init__(self, X_train, y_train):

        self._X_data = None
        self._y_data = None
        self._n_features = None

        self.shared_X = shared(X_train)
        self.shared_y = shared(y_train)

