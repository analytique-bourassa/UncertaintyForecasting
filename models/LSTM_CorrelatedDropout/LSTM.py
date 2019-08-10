import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable
import numpy as np


def loss_correlated_dropout():
    raise NotImplementedError
    return 0

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2, dropout=0, bidirectional=False):

        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.number_of_directions = 1 + 1*self.bidirectional

        Sigma_numpy = np.random.random_integers(0.05, 0.1, size=(self.hidden_dim, self.hidden_dim))
        Sigma_symmetric_numpy = (Sigma_numpy + Sigma_numpy.T) / 2

        self.sigma_matrix_not_activated = Variable(torch.from_numpy(Sigma_symmetric_numpy).double(),
                                     requires_grad=False)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim*self.number_of_directions, output_dim)

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

        self.sigma_matrix = F.relu(self.sigma_matrix_not_activated)

        #self.sigma_matrix = self.sigma_matrix_not_activated
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        if self.training:

            weights = self.linear.weight
            bias = self.linear.bias

            noise_generator = Normal(loc=0, scale=1)
            random_vector = noise_generator.rsample((self.hidden_dim,)).double()

            eigenvalues, eigenvectors = torch.symeig(self.sigma_matrix, eigenvectors=True)
            deviation = torch.mm(torch.mul(torch.sqrt(eigenvalues), random_vector).unsqueeze(0), eigenvectors)
            new_weights = weights.float() + deviation.float()


            y_pred = F.linear(lstm_out[-1].view(self.batch_size, -1), new_weights, bias).view(-1)

        else:

            number_of_samples = 100
            y_pred = torch.ones((self.batch_size, number_of_samples))

            #weights = self.linear.weight
            #bias = self.linear.bias

            for sample_index in range(number_of_samples):
                weights = self.linear.weight
                bias = self.linear.bias

                noise_generator = Normal(loc=0, scale=1)
                random_vector = noise_generator.rsample((self.hidden_dim,)).double()

                eigenvalues, eigenvectors = torch.symeig(self.sigma_matrix, eigenvectors=True)
                deviation = torch.mm(torch.mul(torch.sqrt(eigenvalues), random_vector).unsqueeze(0), eigenvectors)
                new_weights = weights.float() + deviation.float()

                output_value = F.linear(lstm_out[-1].view(self.batch_size, -1), new_weights, bias)

                y_pred[:, sample_index] = output_value.reshape(-1)

        return y_pred
