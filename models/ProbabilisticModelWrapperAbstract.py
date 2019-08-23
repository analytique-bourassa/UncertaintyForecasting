import logging
import pymc3 as pm

from abc import ABCMeta
from pymc3.backends.base import MultiTrace
from utils.validator import Validator

class ProbabilisticModelWrapperAbstract:

    __metaclass__ = ABCMeta

    def __init__(self):

        self._show_progress = True
        self._model = None

    @property
    def show_progress(self):
        return self._show_progress

    @show_progress.setter
    def show_progress(self, value):
        Validator.check_type(value, bool)

        self._show_progress = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        Validator.check_type(value, pm.Model)

        self._model = value

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, value):
        Validator.check_type(value, MultiTrace)

        self._trace = value


    def turn_logging_off(self):
        logger = logging.getLogger('pymc3')
        logger.setLevel(logging.ERROR)

        self.show_progress = False

    def calculate_widely_applicable_information_criterion(self):
        widely_applicable_information_criterion_class = pm.waic(self.trace, self.classification_model_2)
        return widely_applicable_information_criterion_class.WAIC

    def sample(self):
        raise NotImplementedError

    def show_trace(self):
        raise NotImplementedError

    def make_predictions(self):
        raise NotImplementedError
