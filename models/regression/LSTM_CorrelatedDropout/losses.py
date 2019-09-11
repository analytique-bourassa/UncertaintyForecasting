import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.validator import Validator

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

        total_loss = (1 / size_of_batch) * (loss_term_from_variational_approximation - loss_term_from_prior)
        total_loss -= loss_term_from_likelihood

        return total_loss

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

    def calculate_loss_term_from_variation_distribution(self, noisy_weights, mu_weights, sigma_weights):

        number_of_weights = noisy_weights.shape[0]

        log_prob_vector = Variable(torch.ones((number_of_weights,)).cuda())
        for i in range(number_of_weights):
            normal_function_for_weights = Normal(mu_weights[i].cuda(), sigma_weights[i].cuda())
            log_prob_vector[i] = normal_function_for_weights.log_prob(noisy_weights[i].cuda())

        loss_term_from_variational_approximation = log_prob_vector.sum()

        return loss_term_from_variational_approximation



class LossRegressionGaussianWithCorrelations(nn.Module):

    def __init__(self, prior_sigma, number_of_weights, use_gpu=True):
        super(LossRegressionGaussianWithCorrelations, self).__init__()

        assert torch.cuda.is_available(), "The current loss implementation need cuda"

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

    def forward(self, noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction, y_true):

        Validator.check_torch_matrix_is_positive(sigma_matrix_weights)
        Validator.check_matching_dimensions(len(noisy_weights), len(mu_weights))
        Validator.check_matching_dimensions(len(mu_weights), sigma_matrix_weights.shape[0])

        Validator.check_is_torch_tensor_single_value(sigma_prediction)
        Validator.check_matching_dimensions(len(y_true), len(mu_prediction))

        if self.use_gpu:
            move_to_gpu(noisy_weights)
            move_to_gpu(mu_weights)
            move_to_gpu(sigma_matrix_weights)
            move_to_gpu(mu_prediction)
            move_to_gpu(sigma_prediction)
            move_to_gpu(y_true)

        size_of_batch = len(y_true)

        # Loss term from prior
        loss_term_from_prior = self.prior_function.log_prob(noisy_weights).sum()

        # Loss term from likelihood
        loss_term_from_likelihood = self.calculate_likelihood_loss_term(y_true, mu_prediction, sigma_prediction)

        # Loss term from variational distribution
        loss_term_from_variational_approximation = self.calculate_loss_term_from_variation_distribution(noisy_weights,
                                                                                                        mu_weights,
                                                                                                        sigma_matrix_weights)

        total_loss = (1/size_of_batch)*(loss_term_from_variational_approximation - loss_term_from_prior)
        total_loss -= loss_term_from_likelihood

        return total_loss

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

    def calculate_loss_term_from_variation_distribution(self, noisy_weights, mu_weights, sigma_matrix_weights):

        normal_function_for_weights = MultivariateNormal(mu_weights.cuda(),
                                                         covariance_matrix=sigma_matrix_weights.cuda())

        loss_term_from_variational_approximation = normal_function_for_weights.log_prob(noisy_weights.cuda())

        return loss_term_from_variational_approximation

def move_to_gpu(variable):

    assert isinstance(variable, torch.Tensor)

    variable = variable.cuda()