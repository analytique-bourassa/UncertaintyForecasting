class ExperimentParameters():

    def __init__(self):

        self.path = "/home/louis/Documents/ConsultationSimpliphAI/" \
               "AnalytiqueBourassaGit/UncertaintyForecasting/models/LSTM_BayesRegressor/.models/"

        self.version = "v0.0.4"
        self.show_figures = True
        self.smoke_test = False
        self.train_lstm = True
        self.save_lstm = False
        self.type_of_data = "sin"  # options are sin or ar5
        self.name = "feature_extractor"

    @property
    def