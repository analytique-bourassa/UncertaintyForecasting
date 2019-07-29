class BayesianLinearRegressionParameters():

    def __init__(self, SMOKE_TEST=False):

        self.number_of_samples_for_predictions = 1000 if not SMOKE_TEST else 1
        self.number_of_samples_for_posterior = 10000 if not SMOKE_TEST else 1
        self.number_of_tuning_steps = 1000 if not SMOKE_TEST else 1
        self.number_of_iterations = 500000 if not SMOKE_TEST else 1






