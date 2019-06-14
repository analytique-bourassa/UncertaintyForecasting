import matplotlib.pyplot as plt
from models.calibration.diagnostics import show_empirical_cdf, show_in_intervals,show_marginal_calibration


#####################
# Set parameters
#####################
n_data = 300



import numpy as np
import statsmodels.api as sm

np.random.seed(12345)
arparams = np.array([.3, -.2, 0.2, 0.1])
maparams = np.array([.65, .35])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag
y = sm.tsa.arma_generate_sample(ar, ma, n_data)

plt.plot(y)
plt.show()





#####################
# Generate data_handling
#####################

data = np.sin(0.2*np.linspace(0, 200, n_data)) + np.random.normal(0,0.1,n_data) #y#
y_true = data
########################
# Models predictions
#######################

n_samples = 100
ideal_forecaster = np.zeros((n_samples, n_data))
unfocused_forecaster = np.zeros((n_samples, n_data))
asymetric_bias_forecaster = np.zeros((n_samples, n_data))

bias = 0.1
for index in range(n_samples):
    ideal_forecaster[index] = np.sin(0.2*np.linspace(0, 200, n_data)) + np.random.normal(0,0.1, n_data)
    unfocused_forecaster[index] = np.sin(0.2 * np.linspace(0, 200, n_data)) \
                                  + bias*(2*np.random.randint(0, 2, size=(n_data))-1) + np.random.normal(0, 0.1, n_data)

    asymetric_bias_forecaster[index] = np.sin(0.2 * np.linspace(0, 200, n_data))
    asymetric_bias_forecaster[index] += bias*np.sign(asymetric_bias_forecaster[index]) + np.random.normal(0, 0.1, n_data)

ideal_forecaster = np.swapaxes(ideal_forecaster, axis1=0, axis2=1)
unfocused_forecaster = np.swapaxes(unfocused_forecaster, axis1=0, axis2=1)
asymetric_bias_forecaster = np.swapaxes(asymetric_bias_forecaster, axis1=0, axis2=1)

x = range(asymetric_bias_forecaster.shape[0])

y_mean = np.mean(ideal_forecaster, axis=1)
error = np.std(ideal_forecaster, axis=1)

plt.plot(x, y_mean, label="Preds")
plt.plot(x, y_true, label="Data")
plt.fill_between(x, y_mean - error, y_mean + error, alpha=0.5)
plt.legend()
plt.show()


y_mean = np.mean(unfocused_forecaster, axis=1)
error = np.std(unfocused_forecaster, axis=1)

plt.plot(x, y_mean, label="Preds")
plt.plot(x, y_true, label="Data")
plt.fill_between(x, y_mean - error, y_mean + error, alpha=0.5)
plt.legend()
plt.show()

y_mean = np.mean(asymetric_bias_forecaster, axis=1)
error = np.std(asymetric_bias_forecaster, axis=1)

plt.plot(x, y_mean, label="Preds")
plt.plot(x, y_true, label="Data")
plt.fill_between(x, y_mean - error, y_mean + error, alpha=0.5)
plt.legend()
plt.show()

deviation_score_probabilistic_calibration = show_in_intervals(ideal_forecaster, y_true)
deviation_score_exceedance_calibration = show_empirical_cdf(ideal_forecaster, y_true)
deviation_score_marginal_calibration = show_marginal_calibration(ideal_forecaster, y_true)

print(" deviation_score_probabilistic_calibration: %.2f "% deviation_score_probabilistic_calibration)
print(" deviation_score_exceedance_calibration: %.2f "% deviation_score_exceedance_calibration)
print(" deviation_score_marginal_calibration: %.2f "% deviation_score_marginal_calibration)

deviation_score_probabilistic_calibration = show_in_intervals(unfocused_forecaster, y_true)
deviation_score_exceedance_calibration = show_empirical_cdf(unfocused_forecaster, y_true)
deviation_score_marginal_calibration = show_marginal_calibration(unfocused_forecaster, y_true)

print(" deviation_score_probabilistic_calibration: %.2f "% deviation_score_probabilistic_calibration)
print(" deviation_score_exceedance_calibration: %.2f "% deviation_score_exceedance_calibration)
print(" deviation_score_marginal_calibration: %.2f "% deviation_score_marginal_calibration)


deviation_score_probabilistic_calibration = show_in_intervals(asymetric_bias_forecaster, y_true)
deviation_score_exceedance_calibration = show_empirical_cdf(asymetric_bias_forecaster, y_true)
deviation_score_marginal_calibration = show_marginal_calibration(asymetric_bias_forecaster, y_true)

print(" deviation_score_probabilistic_calibration: %.2f "% deviation_score_probabilistic_calibration)
print(" deviation_score_exceedance_calibration: %.2f "% deviation_score_exceedance_calibration)
print(" deviation_score_marginal_calibration: %.2f "% deviation_score_marginal_calibration)