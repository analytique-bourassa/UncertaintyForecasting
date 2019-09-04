import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


from models.LSTM_CorrelatedDropout.distribution_tools import initialize_covariance_matrix


class LSTM_correlated_dropout(nn.Module):

    def __init__(self, lstm_params):

        super(LSTM_correlated_dropout, self).__init__()

        self.input_dim = lstm_params.input_dim
        self.hidden_dim = lstm_params.hidden_dim
        self.batch_size = lstm_params.batch_size
        self.num_layers = lstm_params.num_layers
        self.bidirectional = lstm_params.bidirectional
        self.number_of_directions = 1 + 1*self.bidirectional

        dimension = self.hidden_dim + 1
        number_of_elements_of_triangular_matrix = int(dimension * (dimension + 1) / 2)

        factor_covariance_matrix_temp = torch.zeros(dimension, dimension)

        lower_bound = 0.1
        upper_bound = 0.2

        distribution = Uniform(torch.Tensor([lower_bound]), torch.Tensor([upper_bound]))
        vector = distribution.sample(torch.Size([number_of_elements_of_triangular_matrix]))

        factor_covariance_matrix_temp[torch.triu(torch.ones(dimension, dimension)) == 1] = vector.view(-1)

        self.covariance_factor = torch.tensor(factor_covariance_matrix_temp, requires_grad=True, device="cuda")

        self.prediction_sigma = torch.ones(1, requires_grad=True)
        self.weights_mu = torch.zeros(self.hidden_dim + 1, requires_grad=True,  device="cuda:0")

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            dropout=lstm_params.dropout,
                            bidirectional=self.bidirectional)

        # Define the output layer
        #self.linear = nn.Linear(self.hidden_dim*self.number_of_directions, lstm_params.output_dim)

    def __del__(self):

        torch.cuda.empty_cache()

    @property
    def covariance_matrix(self):
        return torch.mm(self.covariance_factor.transpose(0, 1), self.covariance_factor)

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

            deviation = self.generate_correlated_dropout_noise()
            noisy_weights = self.weights_mu.cuda() + deviation.cuda()

            y_pred = F.linear(lstm_out[-1].view(self.batch_size, -1), noisy_weights[:,:-1], noisy_weights[:,-1]).view(-1)

            return noisy_weights.view(-1).clone(), self.weights_mu.view(-1).clone(), \
                    self.covariance_matrix.clone(), y_pred, self.prediction_sigma.clone()

        else:

            number_of_samples = 100
            y_pred = torch.ones((self.batch_size, number_of_samples))

            for sample_index in range(number_of_samples):

                deviation = self.generate_correlated_dropout_noise()
                noisy_weights = self.weights_mu.cuda() + deviation.cuda()
                output_value = F.linear(lstm_out[-1].view(self.batch_size, -1), noisy_weights[:,:-1], noisy_weights[:,-1])

                y_pred[:, sample_index] = output_value.reshape(-1)

            return y_pred

    def generate_correlated_dropout_noise(self):

        noise_generator = Normal(loc=0, scale=1)
        random_vector = noise_generator.rsample((self.hidden_dim + 1,)).cuda().float() # +1 for bias
        eigenvalues, eigenvectors = torch.symeig(self.covariance_matrix, eigenvectors=True)
        deviation = torch.mm(torch.mul(torch.sqrt(eigenvalues), random_vector).unsqueeze(0), eigenvectors)

        return deviation.cuda()

    def show_summary(self):

        print("weights mu: ", self.weights_mu)
        print("covariance", self.covariance_matrix)
        print("sigma", self.prediction_sigma)


