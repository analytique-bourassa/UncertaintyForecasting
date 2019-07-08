import torch
import pyro.distributions as dist
import torch.nn as nn
import pyro


class BayesianLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2, dropout=0, bidirectional=False):
        super(LSTM_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.number_of_directions = 1 + 1 * self.bidirectional

        self.latent_dim = 10
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            self.num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.number_of_directions, output_dim)

        self.fc21 = nn.Linear(self.hidden_dim * self.number_of_directions, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim * self.number_of_directions, self.latent_dim)

        self.softplus = nn.Softplus()

        self.linear_predictor = nn.Linear(self.latent_dim, output_dim)

    def __del__(self):
        torch.cuda.empty_cache()


    def model(self, x_data, y_data):
        pyro.module("decoder", self.decoder)

        with pyro.plate("data", x_data.shape[0]):
            z_loc = x_data.new_zeros(torch.Size((x_data.shape[0], self.z_dim)))
            z_scale = x_data.new_ones(torch.Size((x_data.shape[0], self.z_dim)))

            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            prediction = self.linear_predictor(z)

            pyro.sample("obs", prediction.to_event(1), obs=y_data)

    def guide(self):
        raise NotImplementedError