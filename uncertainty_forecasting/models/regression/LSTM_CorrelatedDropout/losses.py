import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from uncertainty_forecasting.utils.validator import Validator

from uncertainty_forecasting.utils.time_profiler_logging import TimeProfilerLogger
from time_profile_logger.timers import TimerContext, timer_decorator

logger = TimeProfilerLogger.getInstance()

class LossRegressionGaussianNoCorrelations(nn.Module):

    def __init__(self, prior_sigma, number_of_weights, use_gpu=True):

        assert torch.cuda.is_available(), "The current loss implementation need cuda"

        super(LossRegressionGaussianNoCorrelations, self).__init__()

        self.prior_function = Normal(0, prior_sigma)
        self.number_of_weights = number_of_weights
        self._use_gpu = use_gpu

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        Validator.check_type(value, bool)

        self._use_gpu = value

    def forward(self, noisy_weights, mu_weights, sigma_weights, mu_prediction, sigma_prediction, y_true):

        Validator.check_matching_dimensions(len(noisy_weights), len(mu_weights))
        Validator.check_matching_dimensions(len(mu_weights), len(sigma_weights))
        Validator.check_torch_vector_is_positive(sigma_weights)

        Validator.check_is_torch_tensor_single_value(sigma_prediction)
        Validator.check_matching_dimensions(len(y_true), len(mu_prediction))

        size_of_batch = len(y_true)

        # Loss term from prior
        loss_term_from_prior = self.prior_function.log_prob(noisy_weights).sum()

        # Loss term from likelihood
        loss_term_from_likelihood = self.calculate_likelihood_loss_term(y_true, mu_prediction, sigma_prediction)

        # Loss term from variational distribution
        loss_term_from_variational_approximation = self.calculate_loss_term_from_variation_distribution(noisy_weights, mu_weights, sigma_weights)

        total_loss = (loss_term_from_variational_approximation - loss_term_from_prior)#(1 / size_of_batch) *
        total_loss -= loss_term_from_likelihood

        return total_loss

    @timer_decorator(show_time_elapsed=False, logger=logger)
    def calculate_likelihood_loss_term(self, y_true, mu_prediction, sigma_prediction):

        size_of_batch = len(y_true)
        log_prob_vector = Variable(torch.ones((size_of_batch,)).cuda()) \
            if self.use_gpu else Variable(torch.ones((size_of_batch,)))

        for i in range(size_of_batch):
            mu = mu_prediction[i].cuda() if self.use_gpu else mu_prediction[i]
            sigma = sigma_prediction.cuda() if self.use_gpu else sigma_prediction
            y_expected = y_true[i].cuda() if self.use_gpu else y_true[i]

            normal_function_for_likelihood = Normal(mu, sigma)
            log_prob_vector[i] = normal_function_for_likelihood.log_prob(y_expected)

        log_prob_vector = log_prob_vector.cuda()
        loss_term_from_likelihood = log_prob_vector.sum()

        return loss_term_from_likelihood

    @timer_decorator(show_time_elapsed=False, logger=logger)
    def calculate_loss_term_from_variation_distribution(self, noisy_weights, mu_weights, sigma_weights):

        number_of_weights = noisy_weights.shape[0]

        log_prob_vector = Variable(torch.ones((number_of_weights,)).cuda())
        for i in range(number_of_weights):
            normal_function_for_weights = Normal(mu_weights[i].cuda(), sigma_weights[i].cuda())
            log_prob_vector[i] = normal_function_for_weights.log_prob(noisy_weights[i].cuda())

        loss_term_from_variational_approximation = log_prob_vector.sum()

        return loss_term_from_variational_approximation


class LossRegressionGaussianWithCorrelations(nn.Module):

    INDEX_NOISY_WEIGHTS_FOR_SAMPLES = 0
    INDEX_NOISY_WEIGHTS_FOR_WEIGHTS = 1
    INDEX_MU_PREDICTION_FOR_SAMPLES = 1

    def __init__(self, prior_sigma, number_of_weights, use_gpu=True):
        super(LossRegressionGaussianWithCorrelations, self).__init__()

        assert torch.cuda.is_available(), "The current loss implementation need cuda"

        self.prior_function = Normal(0, prior_sigma)
        self.prior_sigma = torch.Tensor([prior_sigma])
        self.number_of_weights = number_of_weights
        self._use_gpu = use_gpu

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        Validator.check_type(value, bool)

        self._use_gpu = value

    def forward(self, noisy_weights,
                      mu_weights,
                      sigma_matrix_weights,
                      mu_prediction,
                      sigma_prediction,
                      y_true):
        """

        variable                    dimensions
        --------                    ----------

        noisy_weights               [n_samples, number_of_weights]
        mu_weights                  [number_of_weights]
        sigma_matrix_weights        [number_of_weights, number_of_weights]
        mu_prediction               [batch_size, n_samples]
        y_true                      [batch_size]

        :param noisy_weights:
        :param mu_weights:
        :param sigma_matrix_weights:
        :param mu_prediction:
        :param sigma_prediction:
        :param y_true:
        :return:
        """


        Validator.check_torch_matrix_is_positive(sigma_matrix_weights)
        number_of_weights = len(mu_weights)
        Validator.check_matching_dimensions(number_of_weights, sigma_matrix_weights.shape[0])

        Validator.check_matching_dimensions(noisy_weights.shape[self.INDEX_NOISY_WEIGHTS_FOR_WEIGHTS],
                                            number_of_weights)

        Validator.check_is_torch_tensor_single_value(sigma_prediction)


        number_of_samples_for_mu_prediction = mu_prediction.shape[1]
        number_of_samples_for_weights = noisy_weights.shape[0]

        Validator.check_matching_dimensions(number_of_samples_for_mu_prediction,
                                            number_of_samples_for_weights)

        if len(noisy_weights.shape) != 2:
            raise ValueError("The expected shape for the noisy weight is [number_of_samples x number_of_weights]")

        if len(mu_prediction.shape) != 2:
            raise ValueError("The expected shape for the mu_prediction is [batch_size x number_of_samples]")

        if self.use_gpu:
            move_to_gpu(noisy_weights)
            move_to_gpu(mu_weights)
            move_to_gpu(sigma_matrix_weights)
            move_to_gpu(mu_prediction)
            move_to_gpu(sigma_prediction)
            move_to_gpu(y_true)

        size_of_batch = len(y_true)

        # Loss term from prior
        loss_term_from_prior = self.calculate_loss_term_from_prior(noisy_weights)

        # Loss term from likelihood
        loss_term_from_likelihood = self.calculate_loss_term_from_likelihood(y_true, mu_prediction, sigma_prediction)

        # Loss term from variational distribution
        loss_term_from_variational_approximation = self.calculate_loss_term_from_variation_distribution(noisy_weights,
                                                                                                        mu_weights,
                                                                                                        sigma_matrix_weights)

        NUMBER_OF_BATCHES = 50
        total_loss = (1/NUMBER_OF_BATCHES)*(loss_term_from_variational_approximation - loss_term_from_prior)
        total_loss -= loss_term_from_likelihood

        return total_loss

    @timer_decorator(show_time_elapsed=False, logger=logger)
    def calculate_loss_term_from_prior(self, noisy_weights):

        number_of_samples = noisy_weights.shape[self.INDEX_NOISY_WEIGHTS_FOR_SAMPLES]

        log_probabilities = self.prior_function.log_prob(noisy_weights.cuda())

        loss_term_from_likelihood = log_probabilities.sum().cuda()/number_of_samples

        return loss_term_from_likelihood

    @timer_decorator(show_time_elapsed=False, logger=logger)
    def calculate_loss_term_from_likelihood(self, y_true, mu_prediction, sigma_prediction):

        size_of_batch = len(y_true)
        log_prob_vector = Variable(torch.ones((size_of_batch,)).cuda()) \
            if self.use_gpu else Variable(torch.ones((size_of_batch,)))

        number_of_samples = mu_prediction.shape[self.INDEX_MU_PREDICTION_FOR_SAMPLES]

        sigma = sigma_prediction.cuda() if self.use_gpu else sigma_prediction

        for index_in_batch in range(size_of_batch):

            mu = mu_prediction[index_in_batch].cuda() if self.use_gpu else mu_prediction[index_in_batch]
            y_expected = y_true[index_in_batch].cuda() if self.use_gpu else y_true[index_in_batch]

            normal_function = Normal(y_expected, sigma)
            log_prob_vector_samples = normal_function.log_prob(mu.cuda())
            log_prob_vector[index_in_batch] = log_prob_vector_samples.sum()

        log_prob_vector = log_prob_vector.cuda()

        loss_term_from_likelihood = log_prob_vector.sum()/number_of_samples

        return loss_term_from_likelihood


    @timer_decorator(show_time_elapsed=False, logger=logger)
    def calculate_loss_term_from_variation_distribution(self, noisy_weights, mu_weights, sigma_matrix_weights):

        number_of_samples = noisy_weights.shape[self.INDEX_NOISY_WEIGHTS_FOR_SAMPLES]
        normal_function_for_weights = MultivariateNormal(mu_weights.cuda(),
                                                             covariance_matrix=sigma_matrix_weights.cuda())

        log_prob_vector = normal_function_for_weights.log_prob(noisy_weights.cuda())
        loss_term_from_variational_approximation = log_prob_vector.sum().cuda() / number_of_samples

        return loss_term_from_variational_approximation


def move_to_gpu(variable):

    assert isinstance(variable, torch.Tensor)

    variable = variable.cuda()