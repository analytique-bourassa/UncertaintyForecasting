import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.calibration.diagnostics import calculate_static_calibration_error

from utils.Timer import Timer
from models.classification.classification_bayesian_softmax_temperature import \
    BayesianSoftmaxClassificationWithTemperatures
from models.classification.classification_bayesian_softmax import BayesianSoftmaxClassification
from models.visualisations import Visualisator

from data_handling.train_test_split import return_train_test_split_indexes

RANDOM_STATE = 3

iris = sns.load_dataset("iris")

x_n = iris.columns[:-1]
x_2 = iris[x_n].values

X = (x_2 - x_2.mean(axis=0)) / x_2.max(axis=0)

data_classes = list(iris['species'].unique())
y = iris['species'].apply(data_classes.index)

number_of_data = len(y)

train_indexes, test_indexes = return_train_test_split_indexes(number_of_data,
                                                              test_size=0.3,
                                                              random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = X[train_indexes], X[test_indexes], \
                               y.values[train_indexes], y.values[test_indexes]

with Timer(name="without_temperature", show_time_when_exit=False) as timer:

    model_without = BayesianSoftmaxClassification(number_of_classes=3,
                                                  number_of_features=4,
                                                  X_train=X_train,
                                                  y_train=y_train)

    model_without.params.number_of_tuning_steps = 5000
    model_without.params.number_of_samples_for_posterior = int(1e5)
    model_without.params.number_of_iterations = int(1e6)

    model_without.sample()
    # model.show_trace()

    predictions = model_without.make_predictions(X_test, y_test)

    y_test_predictions, confidences = predictions.predictions_with_confidence

    accuracy_without_temperatures = 100.0 * accuracy_score(y_test, predictions.predictions)

    curves_without_temperatures, means_per_bin_without_temperatures, deviation_score_without_temperatures = calculate_static_calibration_error(y_test_predictions,
                                                                                                y_test,
                                                                                                confidences,
                                                                                                predictions.number_of_classes)


with Timer(name="with_temperature", show_time_when_exit=False) as timer:

    model_with = BayesianSoftmaxClassificationWithTemperatures(number_of_classes=3,
                                                               number_of_features=4,
                                                               X_train=X_train,
                                                               y_train=y_train)

    model_with.params.number_of_tuning_steps = 5000
    model_with.params.number_of_samples_for_posterior = int(1e5)
    model_with.params.number_of_iterations = int(1e6)

    model_with.sample()
    # model_with.show_trace()

    predictions = model_with.make_predictions(X_test, y_test)

    y_test_predictions, confidences = predictions.predictions_with_confidence

    accuracy_with_temperatures = 100.0 * accuracy_score(y_test, predictions.predictions)

    curves_with_temperatures, means_per_bin_with_temperatures, deviation_score_with_temperatures = calculate_static_calibration_error(y_test_predictions,
                                                                                          y_test,
                                                                                          confidences,
                                                                                          predictions.number_of_classes)


Visualisator.show_calibration_curves(means_per_bin_with_temperatures,
                                     np.array([curves_with_temperatures,
                                      curves_without_temperatures]),
                                     ["with temperatures", "without temperatures"],
                                     title_suffix="(SCE)"
                                     )


print("acc without: {}".format(accuracy_without_temperatures))
print("acc with: {}".format(accuracy_with_temperatures))
print("dev without: {}".format(deviation_score_without_temperatures))
print("dev with: {}".format(deviation_score_with_temperatures))





