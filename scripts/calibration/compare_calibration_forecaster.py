import matplotlib.pyplot as plt
import numpy as np

from uncertainty_forecasting.data_generation.data_generator import return_arma_data, return_sinus_data
from uncertainty_forecasting.models.dummy_forecasters.forecasters import *
from uncertainty_forecasting.models.calibration.analysis import show_analysis

#####################
# Set parameters
#####################
n_data = 300
y = return_arma_data(n_data)
plt.plot(y)
plt.show()

#####################
# Generate data_handling
#####################

mean_values = return_sinus_data(n_data)
data = mean_values + np.random.normal(0, 0.1, n_data) #y#

#+ np.random.normal(0, 0.1, n_data)  # y#
y_true = data



ideal_forecaster = generate_ideal_forecaster_values(mean_values)
unfocused_forecaster = generate_unfocused_forecaster_values(mean_values)
asymetric_bias_forecaster = generate_asymetric_bias_forecaster_values(mean_values)

ideal_forecaster = np.swapaxes(ideal_forecaster, axis1=0, axis2=1)
unfocused_forecaster = np.swapaxes(unfocused_forecaster, axis1=0, axis2=1)
asymetric_bias_forecaster = np.swapaxes(asymetric_bias_forecaster, axis1=0, axis2=1)

show_analysis(ideal_forecaster, y_true, name="ideal forecaster")
show_analysis(unfocused_forecaster, y_true, name="unfocused forecaster forecaster")
show_analysis(asymetric_bias_forecaster, y_true, name="asymmetric bias forecaster forecaster")
