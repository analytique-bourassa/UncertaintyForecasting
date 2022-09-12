import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris = sns.load_dataset("iris")
#y = pd.Categorical(iris['species'])#.labels

x_n = iris.columns[:-1]
x_2 = iris[x_n].values
X = (x_2 - x_2.mean(axis=0))/x_2.std(axis=0)

data_classes = list(iris['species'].unique())
y = iris['species'].apply(data_classes.index)

#indice = list(set(y_2))

with pm.Model() as modelo_s:

    alpha = pm.Normal('alpha', mu=0, sd=10, shape=3)
    beta = pm.Normal('beta', mu=0, sd=10, shape=(4,3))

    mu = alpha + pm.math.dot(X, beta)
    p = pm.math.exp(mu)/pm.math.sum(pm.math.exp(mu), axis=0)

    yl = pm.Categorical('yl', p=p, observed=y)
    #yl = pm.Multinomial('yl', n=1, p=p, observed=y_2)

    #start = pm.find_MAP()
    #step = pm.Metropolis()
    trace_s = pm.sample(1000)
