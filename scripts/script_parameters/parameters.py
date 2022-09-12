from uncertainty_forecasting.data_generation.data_generators_switcher import DatageneratorsSwitcher

class ExperimentParameters():

    def __init__(self):

        self._path = "/home/louis/Documents/ConsultationSimpliphAI/" \
               "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

        self._version = "v0.0.4"
        self._show_figures = True
        self._smoke_test = False
        self._train_lstm = True
        self._save_lstm = False
        self._type_of_data = "sinus"
        self._name = "feature_extractor"

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        assert isinstance(value, str)
        self._path = value

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        assert isinstance(value, str)
        assert value[0] == "v"

        self._version = value

    @property
    def show_figures(self):
        return self._show_figures

    @show_figures.setter
    def show_figures(self, value):
        assert isinstance(value, bool)

        self._show_figures = value

    @property
    def smoke_test(self):
        return self._smoke_test

    @smoke_test.setter
    def smoke_test(self, value):
        assert isinstance(value, bool)

        self._smoke_test = value

    @property
    def train_lstm(self):
        return self._train_lstm

    @train_lstm.setter
    def train_lstm(self, value):
        assert isinstance(value, bool)

        self._train_lstm = value

    @property
    def save_lstm(self):
        return self._save_lstm

    @save_lstm.setter
    def save_lstm(self, value):
        assert isinstance(value, bool)

        self._save_lstm = value

    @property
    def type_of_data(self):
        return self._type_of_data

    @type_of_data.setter
    def type_of_data(self, value):
        assert isinstance(value, str)
        assert value in DatageneratorsSwitcher.POSSIBLE_DATA_TYPE

        self._type_of_data = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        self._name = value
