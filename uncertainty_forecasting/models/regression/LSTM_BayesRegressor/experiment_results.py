import pandas as pd

class ExperimentsResults():

    def __init__(self):

        self.confidence_interval_deviance_score = list()
        self.one_sided_deviance_score = list()
        self.marginal_deviance_score = list()
        self.correlation_score = list()
        self.methods = list()
        self.elapsed_times = list()

    @property
    def dataframe(self):

        dataframe = pd.DataFrame({"Confidence_interval_deviance_score": self.confidence_interval_deviance_score,
                                  "one_sided_deviance_score": self.one_sided_deviance_score,
                                  "marginal_deviance_score": self.marginal_deviance_score,
                                  "correlation_score": self.correlation_score,
                                  "methods": self.methods,
                                  "elaspsed_time": self.elapsed_times
                                  })

        return dataframe

    def save_as_csv(self, path, name):

        self.dataframe.to_csv(path + name + ".csv")

