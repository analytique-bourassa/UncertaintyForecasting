import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from uncertainty_forecasting.models.calibration.diagnostics import calculate_static_calibration_error

from uncertainty_forecasting.utils.timers import TimerContext
from uncertainty_forecasting.utils.time_profiler_logging import TimeProfilerLogger

logger_time = TimeProfilerLogger()

from uncertainty_forecasting.models.classification.classification_bayesian_softmax_temperature import \
    BayesianSoftmaxClassificationWithTemperatures
from uncertainty_forecasting.models.classification.classification_bayesian_softmax import BayesianSoftmaxClassification
from uncertainty_forecasting.visualisations.visualisations import Visualisator

from uncertainty_forecasting.data_handling.train_test_split import return_train_test_split_indexes

SMOKE_TEST = True

iris = sns.load_dataset("iris")

x_n = iris.columns[:-1]
x_2 = iris[x_n].values

X = (x_2 - x_2.mean(axis=0)) / x_2.max(axis=0)

data_classes = list(iris['species'].unique())
y = iris['species'].apply(data_classes.index)

number_of_different_seeds = 50
number_of_data = len(y)

accuracy_without_temperatures = np.zeros(number_of_different_seeds)
accuracy_with_temperatures = np.zeros(number_of_different_seeds)

deviation_without_temperatures = np.zeros(number_of_different_seeds)
deviation_with_temperatures = np.zeros(number_of_different_seeds)

calculation_time_without_temperatures = np.zeros(number_of_different_seeds)
calculation_time_with_temperatures = np.zeros(number_of_different_seeds)

waic_without_temperatures = np.zeros(number_of_different_seeds)
waic_with_temperatures = np.zeros(number_of_different_seeds)

cumulator_has_been_used_in_train_set = np.zeros(number_of_data)

for i in tqdm(range(number_of_different_seeds)):
    train_indexes, test_indexes = return_train_test_split_indexes(number_of_data, test_size=0.3, random_state=i)

    cumulator_has_been_used_in_train_set[train_indexes] += 1

    X_train, X_test, y_train, y_test = X[train_indexes], X[test_indexes], \
                                       y.values[train_indexes], y.values[test_indexes]

    with TimerContext(name="without_temperature",
                      show_time_when_exit=False,
                      logger=logger_time) as timer:

        model_without = BayesianSoftmaxClassification(number_of_classes=3,
                                                      number_of_features=4,
                                                      X_train=X_train,
                                                      y_train=y_train)

        model_without.turn_logging_off()

        model_without.params.number_of_tuning_steps = 5000 if not SMOKE_TEST else 1
        model_without.params.number_of_samples_for_posterior = int(1e5) if not SMOKE_TEST else 1
        model_without.params.number_of_iterations = int(1e6) if not SMOKE_TEST else 1

        model_without.sample()
        # model.show_trace()

        predictions = model_without.make_predictions(X_test, y_test)

        y_test_predictions, confidences = predictions.predictions_with_confidence

        accuracy_without = 100.0 * accuracy_score(y_test, predictions.predictions)

        curves_without, means_per_bin_without, deviation_score = calculate_static_calibration_error(y_test_predictions,
                                                                                                    y_test,
                                                                                                    confidences,
                                                                                                    predictions.number_of_classes)

        accuracy_without_temperatures[i] = accuracy_without
        deviation_without_temperatures[i] = deviation_score
        calculation_time_without_temperatures[i] = timer.elapsed_time()
        waic_without_temperatures[i] = model_without.calculate_widely_applicable_information_criterion()

    with TimerContext(name="with_temperature", show_time_when_exit=False) as timer:

        model_with = BayesianSoftmaxClassificationWithTemperatures(number_of_classes=3,
                                                                   number_of_features=4,
                                                                   X_train=X_train,
                                                                   y_train=y_train)

        model_with.turn_logging_off()

        model_with.params.number_of_tuning_steps = 5000
        model_with.params.number_of_samples_for_posterior = int(1e5)
        model_with.params.number_of_iterations = int(1e6)

        model_with.sample()
        # model_with.show_trace()

        predictions = model_with.make_predictions(X_test, y_test)

        y_test_predictions, confidences = predictions.predictions_with_confidence

        accuracy_with = 100.0 * accuracy_score(y_test, predictions.predictions)

        curves_with, means_per_bin_with, deviation_score = calculate_static_calibration_error(y_test_predictions,
                                                                                              y_test,
                                                                                              confidences,
                                                                                              predictions.number_of_classes)

    # plt.plot(means_per_bin_with, curves_with, ".-")
    # plt.plot(means_per_bin_with, means_per_bin_with)
    # plt.show()

    # deviation = np.abs(curves_with - means_per_bin_with).sum()

        accuracy_with_temperatures[i] = accuracy_with
        deviation_with_temperatures[i] = deviation_score
        calculation_time_with_temperatures[i] = timer.elapsed_time()
        waic_with_temperatures[i] = model_with.calculate_widely_applicable_information_criterion()

print("Mean acc without: {}".format(accuracy_without_temperatures.mean()))
print("Mean acc with: {}".format(accuracy_with_temperatures.mean()))
print("Mean dev without: {}".format(deviation_without_temperatures.mean()))
print("Mean dev with: {}".format(deviation_with_temperatures.mean()))
print("Mean time calculation without: {}".format(calculation_time_without_temperatures.mean()))
print("Mean time calculation with: {}".format(calculation_time_with_temperatures.mean()))
print("Mean WAIC without: {}".format(waic_without_temperatures.mean()))
print("Mean WAIC with: {}".format(waic_with_temperatures.mean()))

dataframe = pd.DataFrame({"accuracy_without_temperatures": accuracy_without_temperatures,
                          "accuracy_with_temperatures": accuracy_with_temperatures,
                          "deviation_without_temperatures": deviation_without_temperatures,
                          "deviation_with_temperatures": deviation_with_temperatures,
                          "calculation_time_without_temperatures": calculation_time_without_temperatures,
                          "calculation_time_with_temperatures": calculation_time_with_temperatures,
                          "WAIC_without_temperatures": waic_without_temperatures,
                          "WAIC_with_temperatures": waic_with_temperatures
                          })

dataframe.to_csv("data_accuracy_and_deviation_score_with_time_and_waic.csv")

Visualisator.show_multiple_distribution(np.array([accuracy_without_temperatures,
                                                  accuracy_with_temperatures]),
                                        labels=["without temperatures", "with temperatures"],
                                        title="Distributions",
                                        variable_name="accuracy")

df_cumulator = pd.DataFrame({"number_of_times_used_for_training": cumulator_has_been_used_in_train_set})
df_cumulator.to_csv("cumulator_with_time_and_waic.csv")
