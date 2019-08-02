import pymc3 as pm
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.classification.classification_bayesian_softmax import BayesianSoftmaxClassification

iris = sns.load_dataset("iris")

x_n = iris.columns[:-1]
x_2 = iris[x_n].values
X = (x_2 - x_2.mean(axis=0))/x_2.max(axis=0)

data_classes = list(iris['species'].unique())
y = iris['species'].apply(data_classes.index)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = BayesianSoftmaxClassification(number_of_classes=3,
                                      number_of_features=4,
                                      X_train=X_train,
                                      y_train=y_train.values)


model.sample()
model.show_trace()
predictions = model.make_predictions(X_test, y_test.values)



print("accuracy %.2f %%" % (100.0*accuracy_score(y_test, predictions)))
