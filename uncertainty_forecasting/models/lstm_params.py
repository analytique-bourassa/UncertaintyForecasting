import json

from uncertainty_forecasting.utils.validator import Validator

class LSTM_parameters():

    def __init__(self):

        self._input_dim = None
        self._hidden_dim = None
        self._batch_size = None
        self._num_layers = None
        self._bidirectional = None
        self._dropout = None
        self._output_dim = None

        self.input_dim = 1
        self.hidden_dim = 5
        self.batch_size = 20
        self.num_layers = 1
        self.bidirectional = False
        self.dropout = 0.4
        self.output_dim = 1

    @property
    def as_dictionary(self):
        dict_params = dict()

        dict_params["input_dim"] = self.input_dim
        dict_params["hidden_dim"] = self.hidden_dim
        dict_params["batch_size"] = self.batch_size
        dict_params["num_layers"] = self.num_layers
        dict_params["bidirectional"] = self.bidirectional
        dict_params["dropout"] = self.dropout
        dict_params["output_dim"] = self.output_dim

        return dict_params

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):

        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._input_dim = value

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @hidden_dim.setter
    def hidden_dim(self, value):
        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._hidden_dim = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._batch_size = value

    @property
    def num_layers(self):
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._num_layers = value

    @property
    def bidirectional(self):
        return self._bidirectional

    @bidirectional.setter
    def bidirectional(self, value):
        Validator.check_type(value, bool)

        self._bidirectional = value

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, value):
        Validator.check_value_is_a_number(value)
        Validator.check_value_is_between_zero_and_one_inclusive(value)

        self._dropout = value

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value):
        Validator.check_type(value, int)
        Validator.check_value_strictly_positive(value)

        self._output_dim = value

    def save(self, filename, path="./"):

        assert isinstance(filename, str)
        assert isinstance(path, str)

        full_filename = path + filename + "_lstm_params" + ".json"

        with open(full_filename, "w") as file:
            json.dump(self.as_dictionary, file, indent=2)


    def load(self, filename, path=None):

        assert isinstance(filename, str)
        assert isinstance(path, str)

        full_filename = path + filename + ".json"

        with open(full_filename, "r") as file:
            dictionary_read = json.loads(file.read())

        self.input_dim = dictionary_read["input_dim"]
        self.hidden_dim = dictionary_read["hidden_dim"]
        self.batch_size = dictionary_read["batch_size"]
        self.num_layers = dictionary_read["num_layers"]
        self.bidirectional = dictionary_read["bidirectional"]
        self.dropout = dictionary_read["dropout"]
        self.output_dim = dictionary_read["output_dim"]

