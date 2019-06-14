import numbers
import torch
import json

class TrainingParameters():

    def __init__(self):

        self._learning_rate = 1e-3
        self._num_epochs = 12000
        self._use_gpu = torch.cuda.is_available()
        self._batch_size = None

        self._checkpoints_intervals = 1000
        self._save_checkpoints_bool = True
        self._checkpoints_saving_path = "./.checkpoints/"
        self._checkpoints_saving_name = None
        self._do_monitor_training_loss = True
        self._do_early_stopping = False
        self._early_stopping_tolerance = 1e-4
        self._early_stopping_n_iteration_before_stop = 100

    def __str__(self):

        string = "\n"
        string += "****************************\n"
        string += "*** Training parameters ****\n"
        string += "****************************\n"
        string += json.dumps(self.as_dict, indent=2)
        string += "****************************\n"

        return string

    @property
    def as_dict(self):

        dict_with_params = dict()
        dict_with_params["learning_rate"] = self._learning_rate
        dict_with_params["num_epochs"] = self._num_epochs
        dict_with_params["batch_size"] = self._batch_size
        dict_with_params["use_gpu"] = self._use_gpu
        dict_with_params["checkpoints_intervals"] = self._checkpoints_intervals
        dict_with_params["save_checkpoints_bool"] = self._save_checkpoints_bool
        dict_with_params["checkpoints_saving_path"] = self._checkpoints_saving_path
        dict_with_params["checkpoints_saving_name"] = self._checkpoints_saving_name
        dict_with_params["do_monitor_training_loss"] = self._do_monitor_training_loss

        return dict_with_params

    def save(self, filename, path="./"):

        assert isinstance(filename, str), "filename must be a string"
        assert isinstance(path, str), "path must be a string"

        full_filename = path + filename + "_training_params" + ".json"

        json_to_save = self.as_dict
        with open(full_filename, "w") as file:
            json.dump(json_to_save, file, indent=2)

    def load(self, filename, path=None):

        assert isinstance(filename, str), "filename must be a string"
        assert isinstance(path, str), "path must be a string"

        full_filename = path + filename

        with open(full_filename, "r") as file:
            dictionary_read = json.loads(file.read())

        self._learning_rate = dictionary_read["learning_rate"]
        self._num_epochs = dictionary_read["num_epochs"]
        self._batch_size = dictionary_read["batch_size"]
        self._use_gpu = dictionary_read["use_gpu"]
        self._checkpoints_intervals = dictionary_read["checkpoints_intervals"]
        self._save_checkpoints_bool = dictionary_read["save_checkpoints_bool"]
        self._checkpoints_saving_path = dictionary_read["checkpoints_saving_path"]
        self._checkpoints_saving_name = dictionary_read["checkpoints_saving_name"]


    @property
    def use_gpu(self):
        return self._use_gpu

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):

        assert value > 0, "The learning rate must be greater than zero"
        assert value < 1, "The learning rate must be lower than one"
        assert isinstance(value,  numbers.Number)

        self._learning_rate = value

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, value):
        assert value > 0, "The number of epochs must be greater than zero"
        assert isinstance(value, int)

        self._num_epochs = value

    @property
    def checkpoints_intervals(self):
        return self._checkpoints_intervals

    @checkpoints_intervals.setter
    def checkpoints_intervals(self, value):
        assert value > 0, "The checkpoints intervals must be greater than zero"
        assert isinstance(value, int)

        self._checkpoints_intervals = value

    @property
    def save_checkpoints_bool(self):
        return self._save_checkpoints_bool

    @save_checkpoints_bool.setter
    def save_checkpoints_bool(self,value):
        assert isinstance(value, bool), "Must be a boolean"

        self._save_checkpoints_bool = value

    @property
    def checkpoints_saving_path(self):
        return self._checkpoints_saving_path

    @checkpoints_saving_path.setter
    def checkpoints_saving_path(self, value):
        assert isinstance(value, str), "Must be a string"
        self._checkpoints_saving_path = value

    @property
    def checkpoints_saving_name(self):
        return self._checkpoints_saving_name

    @checkpoints_saving_name.setter
    def checkpoints_saving_name(self, value):
        assert isinstance(value, str), "Must be a string"
        self._checkpoints_saving_name = value


    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):

        assert isinstance(value, int), "Mut be an integer"
        assert value > 0, "must be greater than zero"

        self._batch_size = value

    @property
    def do_monitor_training_loss(self):
        return self._do_monitor_training_loss

    @do_monitor_training_loss.setter
    def do_monitor_training_loss(self, value):
        assert isinstance(value, bool), "Must be a boolean"

        self._do_monitor_training_loss = value

    """
    @property
    def do_early_stopping(self):
        return self._do_early_stopping

    @do_early_stopping.setter
    def do_early_stopping(self, value):
        assert isinstance(value, bool), "Must be a boolean"

        self._do_early_stopping = value
    
    @property
    def early_stopping_tolerance(self):
        return self._early_stopping_tolerance

    @early_stopping_tolerance.setter
    def early_stopping_tolerance(self, value):
        assert value > 0, "The learning rate must be greater than zero"
        assert value < 1, "The learning rate must be lower than one"
        assert isinstance(value, numbers.Number)

        self._early_stopping_tolerance = value
    """

