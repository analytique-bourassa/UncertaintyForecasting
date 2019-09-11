import pytest
import torch
import numpy as np

from torch.distributions.normal import Normal
from torch.autograd import Variable

from models.regression.LSTM_CorrelatedDropout.distribution_tools import initialize_covariance_matrix
from models.regression.LSTM_CorrelatedDropout.losses import LossRegressionGaussianWithCorrelations
from models.regression.LSTM_CorrelatedDropout.losses import LossRegressionGaussianNoCorrelations


class TestLossWithCorrelation(object):


    def test_given_loss_arguments_should_return_a_number_for_loss(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()
        mu_weights =  noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = initialize_covariance_matrix(number_of_weights)
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianWithCorrelations(0.1, number_of_weights)

        # Action

        loss = loss_function(noisy_weights,mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction, y_true)

        # Assert
        assert len(loss.shape) == 0
        assert loss.device.type == 'cuda'

    def test_given_loss_arguments_batch_size_10_should_return_a_number_for_loss(self):
        # Prepare
        number_of_weights = np.random.randint(1, 20)  # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()
        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = initialize_covariance_matrix(number_of_weights)

        batch_size = 10
        mu_prediction = noise_generator.rsample((batch_size,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(batch_size))

        loss_function = LossRegressionGaussianWithCorrelations(0.1, number_of_weights)

        # Action

        loss = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction, y_true)

        # Assert
        assert len(loss.shape) == 0
        assert loss.device.type == 'cuda'

    def test_given_not_positive_covariance_matrix_should_return_a_value_error(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()
        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = -1*torch.ones((number_of_weights,number_of_weights))
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianWithCorrelations(0.1, number_of_weights)

        # Action
        with pytest.raises(ValueError):
            _ = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction,
                                 y_true)


        # Assert

    def test_given_numpy_covariance_matrix_should_return_a_type_error(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()
        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = -1*np.ones((number_of_weights,number_of_weights))
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianWithCorrelations(0.1, number_of_weights)

        # Action
        with pytest.raises(TypeError):
            _ = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction,
                                 y_true)


        # Assert

    def test_given_too_many_noisy_weights_should_return_a_value_error(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights
                                                 + np.random.randint(1, 5),)).float()

        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = initialize_covariance_matrix(number_of_weights)
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianWithCorrelations(0.1, number_of_weights)

        # Action
        with pytest.raises(ValueError):
            _ = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction,
                                 y_true)


        # Assert

    def test_given_too_big_covariance_matrix_should_return_a_value_error(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()

        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = initialize_covariance_matrix(number_of_weights + np.random.randint(1, 5))
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianWithCorrelations(0.1, number_of_weights)

        # Action
        with pytest.raises(ValueError):
            _ = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction,
                                 y_true)


        # Assert


class TestLossNoCorrelation(object):


    def test_given_loss_arguments_should_return_a_number_for_loss(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()
        mu_weights =  noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = torch.ones(number_of_weights)
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianNoCorrelations(0.1, number_of_weights)

        # Action

        loss = loss_function(noisy_weights,mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction, y_true)

        # Assert
        assert len(loss.shape) == 0
        assert loss.device.type == 'cuda'

    def test_given_loss_arguments_batch_size_23_should_return_a_number_for_loss(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()
        mu_weights =  noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = torch.ones(number_of_weights)

        batch_size = 23
        mu_prediction = noise_generator.rsample((batch_size,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(batch_size))

        loss_function = LossRegressionGaussianNoCorrelations(0.1, number_of_weights)

        # Action

        loss = loss_function(noisy_weights,mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction, y_true)

        # Assert
        assert len(loss.shape) == 0
        assert loss.device.type == 'cuda'

    def test_given_not_positive_variance_vector_should_return_a_value_error(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()
        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = -1*torch.ones(number_of_weights)
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianNoCorrelations(0.1, number_of_weights)

        # Action
        with pytest.raises(ValueError):
            _ = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction,
                                 y_true)


        # Assert

    def test_given_numpy_covariance_matrix_should_return_a_type_error(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()
        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = -1*np.ones(number_of_weights)
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianNoCorrelations(0.1, number_of_weights)

        # Action
        with pytest.raises(TypeError):
            _ = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction,
                                 y_true)


        # Assert


    def test_given_too_many_noisy_weights_should_return_a_value_error(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights
                                                 + np.random.randint(1, 5),)).float()

        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = torch.ones(number_of_weights)
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianNoCorrelations(0.1, number_of_weights)

        # Action
        with pytest.raises(ValueError):
            _ = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction,
                                 y_true)


        # Assert

    def test_given_too_big_covariance_matrix_should_return_a_value_error(self):

        # Prepare
        number_of_weights = np.random.randint(1, 20) # the upper limit is only to restrict the computation time
        noise_generator = Normal(loc=0, scale=1)

        noisy_weights = noise_generator.rsample((number_of_weights,)).float()

        mu_weights = noise_generator.rsample((number_of_weights,)).float()
        sigma_matrix_weights = torch.ones(number_of_weights + np.random.randint(1, 5))
        mu_prediction = noise_generator.rsample((1,)).float().view(-1)
        sigma_prediction = Variable(torch.ones(1))
        y_true = Variable(torch.zeros(1))

        loss_function = LossRegressionGaussianNoCorrelations(0.1, number_of_weights)

        # Action
        with pytest.raises(ValueError):
            _ = loss_function(noisy_weights, mu_weights, sigma_matrix_weights, mu_prediction, sigma_prediction,
                                 y_true)


        # Assert