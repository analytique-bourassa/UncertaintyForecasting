import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from utils.validator import Validator
#from models.regression.LSTM_CorrelatedDropout.distribution_tools import initialize_covariance_matrix
from torch.distributions.multivariate_normal import MultivariateNormal

class LSTM_correlated_dropout(nn.Module):

    def __init__(self, lstm_params, is_pretraining=True):

        assert torch.cuda.is_available(), "The current model implementation need cuda"

        super(LSTM_correlated_dropout, self).__init__()

        self.params = lstm_params
        self.number_of_directions = 1 + 1*self.params.bidirectional

        dimension = self.params.hidden_dim + 1

        lower_bound = 0.3
        upper_bound = 0.8
        n_values = int(dimension*(dimension+1)/2)

        distribution = Uniform(torch.Tensor([lower_bound]), torch.Tensor([upper_bound]))
        values = distribution.sample(torch.Size([n_values, ])).view(-1)
        matrix = torch.zeros(dimension,dimension)
        matrix[torch.triu(torch.ones(dimension, dimension)) == 1] = values

        #normed = self.normalize_cov_factor(matrix)

        self.covariance_factor = matrix.clone().detach().cuda().requires_grad_(True)

        self.prediction_sigma = torch.tensor([0.01], requires_grad=True, device="cuda:0")
        self.weights_mu = torch.zeros(self.params.hidden_dim + 1, requires_grad=True,  device="cuda:0")

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.params.input_dim,
                            self.params.hidden_dim,
                            self.params.num_layers,
                            dropout=self.params.dropout,
                            bidirectional=self.params.bidirectional)


        self.is_pretraining = is_pretraining

    def __del__(self):

        torch.cuda.empty_cache()

    @property
    def covariance_matrix(self):
        M = torch.mm(self.covariance_factor.transpose(0, 1), self.covariance_factor)
        activation = nn.Softplus()
        return 0.1*activation(M).view(self.params.hidden_dim + 1, self.params.hidden_dim +1)

    @property
    def is_pretraining(self):
        return self._is_pretraining

    @is_pretraining.setter
    def is_pretraining(self, value):

        Validator.check_type(value, bool)

        if not value:
            for param in self.lstm.parameters():
                param.requires_grad = False
        else:
            for param in self.lstm.parameters():
                param.requires_grad = True

        self._is_pretraining = value

    def init_hidden(self):

        # This is what we'll initialise our hidden state as
        zeros_1 = torch.zeros(self.params.num_layers*self.number_of_directions,
                              self.params.batch_size, self.params.hidden_dim)

        zeros_2 = torch.zeros(self.params.num_layers*self.number_of_directions,
                              self.params.batch_size, self.params.hidden_dim)

        hidden = (zeros_1,
                  zeros_2)

        return hidden

    def forward(self, input):

        lstm_out, self.hidden = self.lstm(input.view(len(input), self.params.batch_size, -1))

        if self.training:

            if self.is_pretraining:

                y_pred = self.make_linear_product(self.weights_mu, lstm_out)

                return y_pred

            else:
                deviation = self.generate_correlated_dropout_noise()
                noisy_weights = self.weights_mu.cuda() + deviation.cuda()

                y_pred_mean = self.make_linear_product(noisy_weights,lstm_out)
                #y_pred = self.add_noise_to_predictions_means(y_pred_mean)

                return noisy_weights.view(-1).clone(), self.weights_mu.view(-1).clone(), \
                        self.covariance_matrix.clone(), y_pred_mean, self.prediction_sigma.clone()

        else:

            number_of_samples = 100
            y_pred = torch.ones((self.params.batch_size, number_of_samples))

            for sample_index in range(number_of_samples):

                deviation = self.generate_correlated_dropout_noise()
                noisy_weights = self.weights_mu.cuda() + deviation.cuda()

                y_pred_mean = self.make_linear_product(noisy_weights, lstm_out)
                output_value = self.add_noise_to_predictions_means(y_pred_mean)

                y_pred[:, sample_index] = output_value.reshape(-1)

            return y_pred

    def generate_correlated_dropout_noise(self):

        noise_generator = MultivariateNormal(torch.zeros(self.params.hidden_dim + 1).cuda(),
                                             self.covariance_matrix)
        deviation = noise_generator.rsample()

        return deviation.cuda()

    def show_summary(self):

        print("weights mu: ", self.weights_mu)
        print("covariance: ", self.covariance_matrix)
        print("sigma: ", self.prediction_sigma)

    def make_linear_product(self, weights, lstm_out):

        return F.linear(lstm_out[-1].view(self.params.batch_size, -1),
                 weights[None, :-1], weights[None, -1]).view(-1)

    def add_noise_to_predictions_means(self, y_pred_mean):

        noise_generator = Normal(0, self.prediction_sigma)
        noise = noise_generator.rsample(torch.Size([self.params.batch_size, ]))

        return y_pred_mean + noise.view(-1)


