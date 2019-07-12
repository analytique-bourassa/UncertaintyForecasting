import torch
import torch.nn as nn
from models.lstm_params import LSTM_parameters

class Encoder(nn.Module):

    def __init__(self, lstm_params):

        super(Encoder, self).__init__()

        assert isinstance(lstm_params, LSTM_parameters)
        self.params = lstm_params
        self.number_of_directions = 1 + 1*lstm_params.bidirectional

        # Define the LSTM layer
        self.lstm = nn.LSTM(lstm_params.input_dim,
                            lstm_params.hidden_dim,
                            lstm_params.num_layers,
                            dropout=lstm_params.dropout,
                            bidirectional=lstm_params.bidirectional)


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
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.params.batch_size, -1))

        return self.hidden
