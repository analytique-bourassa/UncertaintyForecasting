import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from torch.autograd import Variable
import numpy as np


class LSTM_not_correlated_dropout(nn.Module):

    def __init__(self, lstm_params):

        super(LSTM_not_correlated_dropout, self).__init__()

        self.input_dim = lstm_params.input_dim
        self.hidden_dim = lstm_params.hidden_dim
        self.batch_size = lstm_params.batch_size
        self.num_layers = lstm_params.num_layers
        self.bidirectional = lstm_params.bidirectional
        self.number_of_directions = 1 + 1*self.bidirectional
        self.output_dim = 1

        self.initialize_variance_vector()
        self.prediction_sigma = Variable(torch.ones(1))

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            dropout=lstm_params.dropout,
                            bidirectional=self.bidirectional)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim*self.number_of_directions, self.output_dim)

    def __del__(self):

        torch.cuda.empty_cache()

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        zeros_1 = torch.zeros(self.num_layers*self.number_of_directions, self.batch_size, self.hidden_dim)
        zeros_2 = torch.zeros(self.num_layers*self.number_of_directions, self.batch_size, self.hidden_dim)
        hidden = (zeros_1,
                  zeros_2)

        return hidden

    def forward(self, input):

        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        if self.training:

            weights, bias = self.linear.weight, self.linear.bias
            deviation = self.generate_dropout_noise()
            noisy_weights = weights.float() + deviation.float()
            y_pred = F.linear(lstm_out[-1].view(self.batch_size, -1), noisy_weights, bias).view(-1)

            return noisy_weights.view(-1), weights.view(-1), self.covariance_vector.view(-1), y_pred, self.prediction_sigma


        else:

            number_of_samples = 100
            y_pred = torch.ones((self.batch_size, number_of_samples))

            for sample_index in range(number_of_samples):

                weights, bias = self.linear.weight, self.linear.bias
                deviation = self.generate_dropout_noise()
                noisy_weights = weights.float() + deviation.float()
                output_value = F.linear(lstm_out[-1].view(self.batch_size, -1), noisy_weights, bias)
                y_pred[:, sample_index] = output_value.reshape(-1)

            return y_pred

    def initialize_variance_vector(self):

        number_of_elements = self.hidden_dim

        lower_bound = 0.1
        upper_bound = 0.2

        distribution = Uniform(torch.Tensor([lower_bound]), torch.Tensor([upper_bound]))
        vector = distribution.rsample((self.hidden_dim,))

        covariance_vector = vector

        self.covariance_vector = covariance_vector

    def generate_dropout_noise(self):

        noise_generator = Normal(loc=0, scale=1)
        noise_generated = noise_generator.rsample((self.hidden_dim,)).double()
        random_vector = torch.mul(noise_generated, self.covariance_vector.view(-1).double())

        return random_vector.cuda().float()


