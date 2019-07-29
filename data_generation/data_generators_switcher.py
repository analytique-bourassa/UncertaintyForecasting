import numpy as np
from data_generation.data_generator import *

class DatageneratorsSwitcher():

    KEY_FOR_SINUS = 0
    KEY_FOR_ARMA = 1
    POSSIBLE_DATA_TYPE = ["sinus", "autoregressive-5"]

    def __init__(self, type_of_data):

        assert type_of_data in self.POSSIBLE_DATA_TYPE

        self.data =None
        self.type_of_data = type_of_data

    def __call__(self, n_data):

        sigma = 0.1

        if self.type_of_data == self.POSSIBLE_DATA_TYPE[self.KEY_FOR_ARMA]:
            y = return_arma_data(n_data)

        elif self.type_of_data == self.POSSIBLE_DATA_TYPE[self.KEY_FOR_SINUS]:
            y = np.sin(0.2 * np.linspace(0, n_data, n_data)) + np.random.normal(0, sigma, n_data)

        else:
            raise ValueError("invalid type of data")

        data = y
        data /= data.max()

        return data

