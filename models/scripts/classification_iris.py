import pymc3 as pm
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models.calibration.diagnostics import calculate_static_calibration_error
import matplotlib.pyplot as plt

from models.classification.classification_bayesian_softmax_temperature import \
    BayesianSoftmaxClassificationWithTemperatures
from models.classification.classification_bayesian_softmax import BayesianSoftmaxClassification

iris = sns.load_dataset("iris")


x_n = iris.columns[:-1]
x_2 = iris[x_n].values

X = (x_2 - x_2.mean(axis=0)) / x_2.max(axis=0)

data_classes = list(iris['species'].unique())
y = iris['species'].apply(data_classes.index)

number_of_different_seeds = 1

accuracy_without_temperatures = np.zeros(number_of_different_seeds)
accuracy_with_temperatures = np.zeros(number_of_different_seeds)

deviation_without_temperatures = np.zeros(number_of_different_seeds)
deviation_with_temperatures = np.zeros(number_of_different_seeds)

for i in tqdm(range(number_of_different_seeds)):
    print("Seed: %d " % i)
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.3, random_state=i)

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

    accuracy_without = 100.0 * accuracy_score(y_test, predictions.predictions)

    curves_without, means_per_bin_without = calculate_static_calibration_error(y_test_predictions,
                                                                               y_test,
                                                                               confidences,
                                                                               predictions.number_of_classes)

    deviation = np.abs(curves_without - means_per_bin_without).sum()

    accuracy_without_temperatures[i] = accuracy_without
    deviation_without_temperatures[i] = deviation

    model_with = BayesianSoftmaxClassificationWithTemperatures(number_of_classes=3,
                                                               number_of_features=4,
                                                               X_train=X_train,
                                                               y_train=y_train)

    model_with.params.number_of_tuning_steps = 5000
    model_with.params.number_of_samples_for_posterior = int(1e5)
    model_with.params.number_of_iterations = int(1e6)

    model_with.sample()
    model_with.show_trace()

    predictions = model_with.make_predictions(X_test, y_test)

    y_test_predictions, confidences = predictions.predictions_with_confidence

    accuracy_with = 100.0 * accuracy_score(y_test, predictions.predictions)

    curves_with, means_per_bin_with = calculate_static_calibration_error(y_test_predictions,
                                                                         y_test,
                                                                         confidences,
                                                                         predictions.number_of_classes)

    plt.plot(means_per_bin_with, curves_with, ".-")
    plt.plot(means_per_bin_with, means_per_bin_with)
    plt.show()

    deviation = np.abs(curves_with - means_per_bin_with).sum()

    accuracy_with_temperatures[i] = accuracy_with
    deviation_with_temperatures[i] = deviation

print("Mean acc without: {}".format(accuracy_without_temperatures.mean()))
print("Mean acc with: {}".format(accuracy_with_temperatures.mean()))
print("Mean dev without: {}".format(deviation_without_temperatures.mean()))
print("Mean dev with: {}".format(deviation_with_temperatures.mean()))

dataframce = pd.DataFrame({"accuracy_without_temperatures":accuracy_without_temperatures,
                           "accuracy_with_temperatures": accuracy_with_temperatures,
                            "deviation_without_temperatures": deviation_without_temperatures,
                            "deviation_with_temperatures": deviation_with_temperatures})


