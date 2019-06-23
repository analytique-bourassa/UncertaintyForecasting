from abc import ABCMeta

class GaussianLinearModel_abstract:

    __metaclass__ = ABCMeta

    def sample(self):
        raise NotImplementedError

    def show_trace(self):
        raise NotImplementedError

    def make_predictions(self):
        raise NotImplementedError
