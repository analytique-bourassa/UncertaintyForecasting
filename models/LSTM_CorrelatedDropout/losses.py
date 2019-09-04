import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.validator import Validator

class LossRegressionGaussianNoCorrelations(nn.Module):

    def __init__(self, prior_sigma, number_of_weights):
        super(LossRegressionGaussianNoCorrelations, self).__init__()

        self.prior_function = Normal(0, prior_sigma)
        self.number_of_weights = number_of_weights

    def forward(self, noisy_weights, mu_weights, sigma_weights, mu_prediction, sigma_prediction, y_true):

        Validator.check_matching_dimensions(len(noisy_weights), len(mu_weights))
        Validator.check_matching_dimensions(len(mu_weights), len(sigma_weights))
        Validator.check_torch_vector_is_positive(sigma_weights)

        Validator.check_is_torch_tensor_single_value(y_true)
        Validator.check_is_torch_tensor_single_value(sigma_prediction)
        Validator.check_is_torch_tensor_single_value(mu_prediction)

        # Loss term from prior
        loss_term_from_prior = self.prior_function.log_prob(noisy_weights).sum()

        # Loss term from likelihood
        normal_function_for_prediction = Normal(mu_prediction.cuda(), sigma_prediction.cuda())
        y = y_true.cuda()
        loss_term_from_likelihood = normal_function_for_prediction.log_prob(y)

        # Loss term from variational distribution
        number_of_weights = noisy_weights.shape[0]

        log_prob_vector = Variable(torch.ones((number_of_weights,)).cuda())
        for i in range(number_of_weights):
            normal_function_for_weights = Normal(mu_weights[i].cuda(), sigma_weights[i].cuda())
            log_prob_vector[i] = normal_function_for_weights.log_prob(noisy_weights[i].cuda())

        loss_term_from_variational_approximation = log_prob_vector.sum()

        total_loss = loss_term_from_variational_approximation - loss_term_from_prior - loss_term_from_likelihood

        return total_loss


class LossRegressionGaussianWithCorrelations(nn.Module):

    def __init__(self, prior_sigma, number_of_weights):
        super(LossRegressionGaussianWithCorrelations, self).__init__()

        self.prior_function = Normal(0, prior_sigma)
        self.number_of_weights = number_of_weights

    def forward(self, noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction, y_true):

        Validator.check_torch_matrix_is_positive(sigma_matrix_weights)
        Validator.check_matching_dimensions(len(noisy_weights), len(mu_weights))
        Validator.check_matching_dimensions(len(mu_weights), sigma_matrix_weights.shape[0])

        Validator.check_is_torch_tensor_single_value(y_true)
        Validator.check_is_torch_tensor_single_value(sigma_prediction)
        Validator.check_is_torch_tensor_single_value(mu_prediction)

        # Loss term from prior
        loss_term_from_prior = self.prior_function.log_prob(noisy_weights.cuda()).sum()

        # Loss term from likelihood
        normal_function_for_prediction = Normal(mu_prediction.cuda(), sigma_prediction.cuda())
        loss_term_from_likelihood = normal_function_for_prediction.log_prob(y_true.cuda())

        # Loss term from variational distribution
        normal_function_for_weights = MultivariateNormal(mu_weights.cuda(), sigma_matrix_weights.cuda())
        loss_term_from_variational_approximation = normal_function_for_weights.log_prob(noisy_weights.cuda())

        total_loss = loss_term_from_variational_approximation - loss_term_from_prior - loss_term_from_likelihood

        return total_loss
