import numpy as np


mu = np.array([1, 2, 3])

betas_1 = np.array([1, 1, 1])
betas_2 = np.array([2, 1, 1])
betas_3 = np.array([0.5, 1, 1])

def softmax(mu, betas):

    factors = np.exp(mu*betas)
    constant = factors.sum()
    probabilities = factors/constant
    return probabilities

print(softmax(mu, betas_1))
print(softmax(mu, betas_2))
print(softmax(mu, betas_3))