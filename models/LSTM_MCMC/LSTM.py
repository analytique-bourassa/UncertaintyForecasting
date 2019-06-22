import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

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
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        #if self.training:
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))

        return y_pred.view(-1)

    def return_last_layer(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        return lstm_out[-1].view(self.batch_size, -1)

    @property
    def last_layers_weights(self):
        return self.linear.weight.view(-1).detach().numpy(), self.linear.bias.detach().numpy()
