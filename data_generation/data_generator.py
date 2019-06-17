import numpy as np
import statsmodels.api as sm

np.random.seed(12345)
arparams = np.array([.3, -.2, 0.2, 0.1])
maparams = np.array([.65, .35])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag


def return_arma_data(n_data):
    return sm.tsa.arma_generate_sample(ar, ma, n_data, sigma=0.0)

def return_sinus_data(n_data):
    return np.sin(0.2 * np.linspace(0, n_data, n_data))





