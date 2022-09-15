import numpy as np
import matplotlib.pyplot as plt

from uncertainty_forecasting.models.calibration.diagnostics import calculate_one_sided_cumulative_calibration, calculate_confidence_interval_calibration, calculate_marginal_calibration

KEY_N_TIMESTEPS = 0

def show_analysis(forecasting_values, y_true, name="ideal forecaster"):

    deviation_score_probabilistic_calibration = calculate_confidence_interval_calibration(forecasting_values, y_true, show=True)
    deviation_score_exceedance_calibration = calculate_one_sided_cumulative_calibration(forecasting_values, y_true, show=True)
    deviation_score_marginal_calibration = calculate_marginal_calibration(forecasting_values, y_true, show=True)

    print(" deviation_score_probabilistic_calibration: %.5f " % deviation_score_probabilistic_calibration)
    print(" deviation_score_exceedance_calibration: %.5f " % deviation_score_exceedance_calibration)
    print(" deviation_score_marginal_calibration: %.5f " % deviation_score_marginal_calibration)