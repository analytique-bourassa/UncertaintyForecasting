import json

class LSTM_parameters():

    def __init__(self):

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


    def save(self, filename, path="./"):

        assert isinstance(filename, str)
        assert isinstance(path, str)

        full_filename = path + filename + "_lstm_params" + ".json"

        with open(full_filename, "w") as file:
            json.dump(self.as_dictionary, file, indent=2)


    def load(self, filename, path=None):

        assert isinstance(filename, str)
        assert isinstance(path, str)

        full_filename = path + filename

        with open(full_filename, "r") as file:
            dictionary_read = json.loads(file.read())

        self.input_dim = dictionary_read["input_dim"]
        self.hidden_dim = dictionary_read["hidden_dim"]
        self.batch_size = dictionary_read["batch_size"]
        self.num_layers = dictionary_read["num_layers"]
        self.bidirectional = dictionary_read["bidirectional"]
        self.dropout = dictionary_read["dropout"]
        self.output_dim = dictionary_read["output_dim"]
