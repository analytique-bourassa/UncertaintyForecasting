import numpy as np
import torch
import itertools

from models.regression.LSTM_CorrelatedDropout.LSTM_correlated_dropout import LSTM_correlated_dropout
from models.regression.LSTM_CorrelatedDropout.losses import LossRegressionGaussianWithCorrelations
from models.lstm_params import LSTM_parameters

from data_generation.data_generators_switcher import DatageneratorsSwitcher
from data_handling.data_reshaping import reshape_data_for_LSTM, reshape_into_sequences
from models.model_data_feeder import *


from utils.validator import Validator
from utils.distances import calculate_Kullback_Leibler_divergence_covariance_matrix


def check_mean_absolute_difference_bigger_than_learning_rate(value_1, value_2, learning_rate):
    return np.mean(np.abs(value_1 - value_2)) >= learning_rate



class TestLSTM_CorrelatedDropout(object):


    def test_if_initialized_covariance_matrix_is_positive(self):

        # Prepare
        lstm_params = LSTM_parameters()

        # Action
        model = LSTM_correlated_dropout(lstm_params)

        # Assert
        Validator.check_torch_matrix_is_positive(model.covariance_matrix)

    def test_if_dropout_generated_has_correlations(self):
        #TODO correct with exact stat test: "An exact test about the covariance matrix"

        # Prepare
        lstm_params = LSTM_parameters()
        model = LSTM_correlated_dropout(lstm_params)
        number_of_sample = 100
        number_of_variables = lstm_params.hidden_dim + 1
        samples = np.zeros((number_of_variables, number_of_sample))
        covariance_expected = model.covariance_matrix.cpu().detach().numpy()

        # Action
        for i in range(number_of_sample):
            samples[:, i] = model.generate_correlated_dropout_noise().cpu().detach().numpy()

        # Assert
        assert calculate_Kullback_Leibler_divergence_covariance_matrix(covariance_expected, np.cov(samples)) <= 1

    def test_if_pretraining_modify_lstm_weights(self):

        def get_lstm_weights(lstm):

            weight_list = list()

            NAME_INDEX = 0
            VALUE_INDEX = 1

            for named_parameter in lstm.named_parameters():
                if 'weight' in named_parameter[NAME_INDEX]:
                    weight_list.append(named_parameter[VALUE_INDEX].cpu().detach().numpy())

            return weight_list


        # Prepare
        n_data = 100
        length_of_sequences = 7
        learning_rate = 1e-3
        num_epochs_pretraining = 20

        data = np.random.normal(0, 1, size=n_data)
        sequences = reshape_into_sequences(data, length_of_sequences)
        data_train = reshape_data_for_LSTM(sequences)

        lstm_params = LSTM_parameters()
        model = LSTM_correlated_dropout(lstm_params)
        model.cuda()

        # Original value used to comparison
        original_weights_lstm = get_lstm_weights(model.lstm)

        # Action
        optimizer_1 = torch.optim.Adam(itertools.chain(model.parameters(), [model.weights_mu]),
                                       lr=learning_rate)

        loss_function_pretraining = torch.nn.MSELoss(size_average=False)

        for epoch in range(num_epochs_pretraining):
            model.hidden = model.init_hidden()

            losses, N_data = make_forward_pass(data_loader_sequences, model, loss_function_pretraining, data_train,
                                               lstm_params.batch_size)

            optimizer_1.zero_grad()
            losses.backward()
            optimizer_1.step()

        # New value
        new_weights_lstm = get_lstm_weights(model.lstm)

        # Assert
        assert len(original_weights_lstm) == len(new_weights_lstm)

        for index_weights in range(len(new_weights_lstm)):

            new_w_lstm = new_weights_lstm[index_weights]
            original_w_lstm = original_weights_lstm[index_weights]

            assert check_mean_absolute_difference_bigger_than_learning_rate(new_w_lstm,
                                                                            original_w_lstm,
                                                                            learning_rate)

    def test_if_training_modify_mu_weights(self):

        # Prepare
        n_data = 100
        length_of_sequences = 7

        data = np.random.normal(0, 1, size=n_data)
        sequences = reshape_into_sequences(data, length_of_sequences)
        data_train = reshape_data_for_LSTM(sequences)

        lstm_params = LSTM_parameters()
        model = LSTM_correlated_dropout(lstm_params)
        model.cuda()
        num_epochs_pretraining = 20
        original_weights_mu = model.weights_mu.cpu().detach().numpy()
        learning_rate = 1e-3

        # Action
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), [model.weights_mu]),
                                       lr=learning_rate)

        loss_function_pretraining = torch.nn.MSELoss(size_average=False)

        for epoch in range(num_epochs_pretraining):
            model.hidden = model.init_hidden()

            losses, N_data = make_forward_pass(data_loader_sequences, model, loss_function_pretraining, data_train,
                                               lstm_params.batch_size)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # New value
        new_weights_mu = model.weights_mu.cpu().detach().numpy()

        # Assert
        assert np.mean(np.abs(new_weights_mu - original_weights_mu)) >= learning_rate

    def test_if_training_modify_sigma_matrix(self):

        # Prepare
        n_data = 100
        length_of_sequences = 7
        num_epochs_pretraining = 20
        learning_rate = 1e-3

        data = np.random.normal(0, 1, size=n_data)
        sequences = reshape_into_sequences(data, length_of_sequences)
        data_train = reshape_data_for_LSTM(sequences)

        lstm_params = LSTM_parameters()
        model = LSTM_correlated_dropout(lstm_params)
        model.cuda()

        # Value used to comparison
        original_covariance_matrix = model.covariance_matrix.cpu().detach().numpy()

        # Action
        optimizer = torch.optim.Adam([model.weights_mu, model.prediction_sigma,
                                                                  model.covariance_factor],
                                       lr=learning_rate)

        loss_training = LossRegressionGaussianWithCorrelations(1.0, lstm_params.hidden_dim)

        model.is_pretraining = False
        for epoch in range(num_epochs_pretraining):
            model.hidden = model.init_hidden()

            losses, N_data = make_forward_pass_output_specific(data_loader_sequences, model, loss_training, data_train,
                                               lstm_params.batch_size)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # New value
        new_covariance_matrix = model.covariance_matrix.cpu().detach().numpy()

        # Assert
        assert check_mean_absolute_difference_bigger_than_learning_rate(original_covariance_matrix,
                                                                        new_covariance_matrix,
                                                                        learning_rate)

    def test_if_training_modify_sigma_prediction(self):

        # Prepare
        n_data = 100
        length_of_sequences = 7
        num_epochs_pretraining = 20
        learning_rate = 1e-3

        data = np.random.normal(0, 1, size=n_data)
        sequences = reshape_into_sequences(data, length_of_sequences)
        data_train = reshape_data_for_LSTM(sequences)

        lstm_params = LSTM_parameters()
        model = LSTM_correlated_dropout(lstm_params)
        model.cuda()

        # Value used to comparison
        original_prediction_sigma = model.prediction_sigma.cpu().detach().numpy()

        # Action
        optimizer = torch.optim.Adam([model.weights_mu, model.prediction_sigma,
                                      model.covariance_factor],
                                     lr=learning_rate)

        loss_training = LossRegressionGaussianWithCorrelations(1.0, lstm_params.hidden_dim)

        model.is_pretraining = False
        for epoch in range(num_epochs_pretraining):
            model.hidden = model.init_hidden()

            losses, N_data = make_forward_pass_output_specific(data_loader_sequences, model, loss_training, data_train,
                                                               lstm_params.batch_size)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # New value
        new_prediction_sigma = model.prediction_sigma.cpu().detach().numpy()

        # Assert
        assert check_mean_absolute_difference_bigger_than_learning_rate(original_prediction_sigma,
                                                                        new_prediction_sigma,
                                                                        learning_rate)
