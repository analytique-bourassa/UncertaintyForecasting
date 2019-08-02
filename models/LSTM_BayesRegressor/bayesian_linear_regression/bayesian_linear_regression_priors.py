from utils.validator import Validator

class BayesianLinearRegressionPriors():

    def __init__(self, priors_thetas=None):

        self._mean_theta_0 = None
        self._mean_thetas = None

        self._standard_deviation_theta_0 = None
        self._standard_deviation_thetas = None

        self._standard_deviation_sigma = None
        self._mean_mu = None

        self.mean_theta_0 = 1.0
        if priors_thetas is not None:
            self.mean_thetas = 1.0

        self.standard_deviation_theta_0 = 2.0
        self.standard_deviation_thetas = 2.0

        self.standard_deviation_sigma = 10.0

    @property
    def mean_theta_0(self):
        return self._mean_theta_0

    @mean_theta_0.setter
    def mean_theta_0(self, value):
        Validator.check_value_is_a_number(value)

        self._mean_theta_0 = value

    @property
    def mean_thetas(self):
        return self._mean_thetas

    @mean_thetas.setter
    def mean_thetas(self, value):
        Validator.check_value_is_a_number(value)

        self._mean_thetas = value

    @property
    def standard_deviation_theta_0(self):
        return self._standard_deviation_theta_0

    @standard_deviation_theta_0.setter
    def standard_deviation_theta_0(self, value):

        Validator.check_value_is_a_number(value)
        Validator.check_value_strictly_positive(value)

        self._standard_deviation_theta_0 = value

    @property
    def standard_deviation_thetas(self):
        return self._standard_deviation_thetas

    @standard_deviation_thetas.setter
    def standard_deviation_thetas(self, value):
        Validator.check_value_is_a_number(value)
        Validator.check_value_strictly_positive(value)

        self._standard_deviation_thetas = value

    @property
    def standard_deviation_sigma(self):
        return self._standard_deviation_sigma

    @standard_deviation_sigma.setter
    def standard_deviation_sigma(self, value):
        Validator.check_value_is_a_number(value)
        Validator.check_value_strictly_positive(value)

        self._standard_deviation_sigma = value


