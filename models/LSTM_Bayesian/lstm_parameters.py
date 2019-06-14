import json
from warnings import warn

from utils.argument_validator import ArgumentValidator


class LSTM_parameters():

    def __init__(self,
                 hidden_layer_1_units=10,
                 output_dim=1,
                 num_layers=1,
                 n_features=1,
                 dropout=0,
                 bidirectional=False,
                 batch_size=100):
        self.h1 = hidden_layer_1_units
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.n_features = n_features
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_size = batch_size

    @property
    def as_dict(self):

        dict_params = dict()
        dict_params["input_dim"] = self.n_features
        dict_params["hidden_dim"] = self.h1
        dict_params["batch_size"] = self.batch_size
        dict_params["output_dim"] = self.output_dim
        dict_params["num_layers"] = self.num_layers
        dict_params["dropout"] = self.dropout
        dict_params["bidirectional"] = self.bidirectional

        return dict_params

    def __str__(self):

        string = "\n"
        string += "**********************************\n"
        string += "****** LSTM parameters ***********\n"
        string += "**********************************\n"
        string += json.dumps(self.as_dict, indent=2)
        string += "\n**********************************\n"

        return string

    def save(self, filename, path="./"):
        ArgumentValidator.check_type(filename, str)
        ArgumentValidator.check_type(path, str)

        full_filename = path + filename + "_lstm_params" + ".json"

        with open(full_filename, "w") as file:
            json.dump(self.as_dict, file, indent=2)

    def load(self, filename, path=None):
        ArgumentValidator.check_type(filename, str)
        ArgumentValidator.check_type(path, str)

        full_filename = path + filename

        with open(full_filename, "r") as file:
            dictionary_read = json.loads(file.read())

        self._hidden_layer_1_units = dictionary_read["hidden_dim"]
        self._output_dim = dictionary_read["output_dim"]
        self._num_layers = dictionary_read["num_layers"]
        self._n_features = dictionary_read["input_dim"]
        self.batch_size = dictionary_read["batch_size"]

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value):
        ArgumentValidator.check_type(value, int)
        if value <= 0:
            raise ValueError("The output_dim must be greater than zero")

        self._output_dim = value

    @property
    def dropout(self):
        if self._dropout > 0 and self.num_layers == 1:
            warn("The dropout is effectively zero because there is only one layer")

        return self._dropout

    @dropout.setter
    def dropout(self, value):
        if value < 0:
            raise ValueError("The dropout must be equal or greater than zero")
        if value >= 1:
            raise ValueError("The dropout must be smaller than 1")

        self._dropout = value

    @property
    def bidirectional(self):
        return self._bidirectional

    @bidirectional.setter
    def bidirectional(self, value):
        ArgumentValidator.check_type(value, bool)
        self._bidirectional = value

    @property
    def h1(self):
        return self._hidden_layer_1_units

    @h1.setter
    def h1(self, value):
        ArgumentValidator.check_type(value, int)
        if value <= 0:
            raise ValueError("The hidden layer 1 number of units must be greater than zero")

        self._hidden_layer_1_units = value

    @property
    def num_layers(self):
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        ArgumentValidator.check_type(value, int)
        if value <= 0:
            raise ValueError("The num_layers must be greater than zero")

        self._num_layers = value

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, value):
        ArgumentValidator.check_type(value, int)
        if value <= 0:
            raise ValueError("The n_features must be greater than zero")

        self._n_features = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        ArgumentValidator.check_type(value, int)
        if value <= 0:
            raise ValueError("batch size must be greater than zero")

        self._batch_size = value

