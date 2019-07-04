import numpy as np
from data_generation.data_generator import *

class DatageneratorsSwitcher():

    POSSIBLE_DATA_TYPE = ["sinus", "autoregressive-5"]

    def __init__(self, type_of_data):

        assert type_of_data in self.POSSIBLE_DATA_TYPE

        self.data =None
        self.type_of_data = type_of_data

    def __call__(self, n_data):

        if self.type_of_data == "autoregressive-5":
            y = return_arma_data(n_data)

        elif self.type_of_data == "sinus":
            y = np.sin(0.2 * np.linspace(0, n_data, n_data))

        else:
            raise ValueError("invalid type of data")

        sigma = 0.1
        data = y + np.random.normal(0, sigma, n_data)
        data /= data.max()

        return data

