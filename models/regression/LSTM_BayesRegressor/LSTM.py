import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from models.lstm_params import LSTM_parameters

class LSTM(nn.Module):

    def __init__(self, lstm_params):

        super(LSTM, self).__init__()

        assert isinstance(lstm_params, LSTM_parameters)
        self.params = lstm_params
        self.number_of_directions = 1 + 1*lstm_params.bidirectional

        self.lstm = nn.LSTM(lstm_params.input_dim,
                            lstm_params.hidden_dim,
                            lstm_params.num_layers,
                            dropout=lstm_params.dropout,
                            bidirectional=lstm_params.bidirectional)

        self.linear = nn.Linear(lstm_params.hidden_dim*self.number_of_directions, lstm_params.output_dim)

    def __del__(self):

        torch.cuda.empty_cache()

    def init_hidden(self):
        zeros_1 = torch.zeros(self.params.num_layers*self.number_of_directions,
                              self.params.batch_size,
                              self.params.hidden_dim)

        zeros_2 = torch.zeros(self.params.num_layers*self.number_of_directions,
                              self.params.batch_size,
                              self.params.hidden_dim)
        hidden = (zeros_1,
                  zeros_2)

        return hidden

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.params.batch_size, -1))
        y_pred = self.linear(lstm_out[-1].view(self.params.batch_size, -1))

        return y_pred.view(-1)

    def return_last_layer(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.params.batch_size, -1))
        return lstm_out[-1].view(self.params.batch_size, -1)

    @property
    def last_layers_weights(self):
        return self.linear.weight.view(-1).detach().numpy(), self.linear.bias.detach().numpy()
