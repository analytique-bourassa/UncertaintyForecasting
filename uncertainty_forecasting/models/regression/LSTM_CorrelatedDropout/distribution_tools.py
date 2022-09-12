import torch
from torch.distributions.uniform import Uniform
from torch.autograd import Variable


def initialize_covariance_matrix(dimension):

    number_of_elements_of_triangular_matrix = int(dimension* (dimension + 1) / 2)

    factor_covariance_matrix = torch.zeros(dimension, dimension)

    lower_bound = 0.1
    upper_bound = 0.2

    distribution = Uniform(torch.Tensor([lower_bound]), torch.Tensor([upper_bound]))
    vector = distribution.sample(torch.Size([number_of_elements_of_triangular_matrix]))

    factor_covariance_matrix[torch.triu(torch.ones(dimension, dimension)) == 1] = vector.view(-1)

    covariance_matrix = torch.mm(factor_covariance_matrix.transpose(0, 1), factor_covariance_matrix)

    return Variable(covariance_matrix.cuda(), requires_grad=True)

