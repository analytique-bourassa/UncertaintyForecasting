import numpy as np

import torch.nn as nn
import torch

from uncertainty_forecasting.models.model_data_feeder import data_loader_sequences
from uncertainty_forecasting.models.model_data_feeder import make_forward_pass
from uncertainty_forecasting.models.model_data_feeder import make_predictions
from uncertainty_forecasting.models.model_data_feeder import extract_features

class TestModelDataFeeder(object):

    def test_data_loader_sequences(self):

        # Prepare
        number_of_data = 1000
        batch_size = 100
        expected_number_of_batches = 10
        number_of_time_steps = 8
        number_of_features = 1

        data_generated = np.random.normal(0, 1, (number_of_time_steps, number_of_data, number_of_features))

        data_generated_by_data_loader = list()

        # Action
        for features, labels in data_loader_sequences(data_generated, batch_size, random=True):
            data_generated_by_data_loader.append(features)

        # Assert
        assert len(data_generated_by_data_loader) == expected_number_of_batches
        assert all([data.shape[1] == batch_size for data in data_generated_by_data_loader])
        assert all([data.shape[0] == number_of_time_steps - 1 for data in data_generated_by_data_loader])

    def test_if_make_forward_pass_return_data(self):


        # Prepare
        number_of_data = 1000
        batch_size = 100
        expected_number_of_batches = 10
        number_of_time_steps = 8
        number_of_features = 1

        data_generated = np.random.normal(0, 1, (number_of_time_steps, number_of_data, number_of_features))

        class EmptyModel(nn.Module):

            def __init__(self):
                super(EmptyModel, self).__init__()
                input_size, hidden_size = 5, 10
                self.lstm = nn.LSTM(input_size, hidden_size)

            def forward(self, x):
                return np.random.normal(0,1, batch_size)


        model = EmptyModel()

        def faked_loss(x, y): return 1

        # Action
        losses, N_data = make_forward_pass(data_loader_sequences,  model, faked_loss, data_generated, batch_size )

        # Assert
        assert N_data == number_of_data
        assert losses == expected_number_of_batches


    def test_if_make_predictions_return_data(self):


        # Prepare
        number_of_data = 1000
        batch_size = 100
        number_of_time_steps = 8
        number_of_features = 1

        data_generated = np.random.normal(0, 1, (number_of_time_steps, number_of_data, number_of_features))

        class EmptyModel(nn.Module):

            def __init__(self):
                super(EmptyModel, self).__init__()
                input_size, hidden_size = 5, 10
                self.lstm = nn.LSTM(input_size, hidden_size)

            def forward(self, x):
                return torch.tensor(np.random.normal(0,1,batch_size))


        model = EmptyModel()


        # Action
        y_pred_all, y_test_all = make_predictions(data_loader_sequences, model, data_generated, batch_size)

        # Assert
        assert len(y_test_all) == number_of_data
        assert len(y_pred_all) == number_of_data


    def test_if_extract_features(self):


        # Prepare
        number_of_data = 1000
        batch_size = 100
        expected_number_of_batches = 10
        number_of_time_steps = 8
        number_of_features = 1
        number_of_hidden_states = 5

        data_generated = np.random.normal(0, 1, (number_of_time_steps, number_of_data, number_of_features))

        class EmptyModel(nn.Module):

            def __init__(self):
                super(EmptyModel, self).__init__()
                input_size, hidden_size = 5, 10
                self.lstm = nn.LSTM(input_size, hidden_size)

            def forward(self, x):
                return torch.tensor(np.random.normal(0,1,batch_size))

            def return_last_layer(self, x):
                return torch.tensor(np.random.normal(0,1,(batch_size,number_of_hidden_states)))


        model = EmptyModel()


        # Action
        features_all, y_test_all = extract_features(data_loader_sequences, model, data_generated, batch_size)

        # Assert
        assert len(y_test_all) == number_of_data
        assert len(features_all) == number_of_data
        assert features_all.shape[1] == number_of_hidden_states
