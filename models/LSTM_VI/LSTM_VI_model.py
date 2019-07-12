from models.LSTM_VI.LSTM import Encoder
import pyro.distributions as dist
import torch.nn as nn
import pyro
import torch
from pyro.contrib.autoguide import AutoDiagonalNormal

class LSTM_VI(nn.Module):

    def __init__(self, params_lstm, use_cuda=False):
        super(LSTM_VI, self).__init__()

        self.encoder = Encoder(params_lstm)
        #self.encoder_deviations = Encoder(params_lstm)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = params_lstm.hidden_dim

    def model(self, x,y):

        w_prior = dist.Normal(torch.zeros(1, self.z_dim), torch.ones(1, self.z_dim)).to_event(1)
        sigma_prior = dist.Uniform(0., 10.)

        w = pyro.sample("weights", w_prior)
        encoding = self.encoder(x)
        mu = torch.mm(encoding, w)

        sigma = pyro.sample("sigma", sigma_prior)

        pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)


    def guide(self):

        return AutoDiagonalNormal(self.model)

