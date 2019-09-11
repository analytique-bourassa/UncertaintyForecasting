import torch
from math import floor
import numpy as np
import torch.nn as nn

from utils.validator import Validator

def data_loader_sequences(data, batch_size, random=True):
    """

    The expected shape is [number_of_times_steps, number_of_data, number_of_features]
    :param data:
    :param batch_size:
    :param random:
    :return:
    """

    Validator.check_type(data, np.ndarray)

    index_for_number_of_data = 1
    number_of_batches = int(floor(data.shape[index_for_number_of_data] / batch_size))

    if random:
        indexes_of_batch = np.random.choice(range(number_of_batches),
                                            number_of_batches, replace=False)
    else:
        indexes_of_batch = range(number_of_batches)

    for index_batch in indexes_of_batch:
        batch_first_index = index_batch * batch_size
        batch_last_index = (index_batch + 1) * batch_size

        X_numpy = data[:-1, batch_first_index:batch_last_index, 0]
        X_torch = torch.from_numpy(X_numpy).type(torch.Tensor)

        y_numpy = data[-1, batch_first_index:batch_last_index, 0]
        y_torch = torch.from_numpy(y_numpy).type(torch.Tensor)

        yield X_torch, y_torch


def make_forward_pass(data_loader, model, loss_fn, training_data, batch_size):

    assert isinstance(model, nn.Module)

    losses = None
    N_data = 0

    for X_train, y_train in data_loader(training_data, batch_size):

        if next(model.parameters()).is_cuda:
            X_train, y_train = X_train.cuda(), y_train.cuda()

        N_data += batch_size
        y_pred = model(X_train)

        if losses is None:
            losses = loss_fn(y_pred, y_train)
        else:
            losses += loss_fn(y_pred, y_train)

    return losses, N_data

def make_forward_pass_output_specific(data_loader, model, loss_fn, training_data, batch_size):

    assert isinstance(model, nn.Module)

    losses = None
    N_data = 0

    for X_train, y_train in data_loader(training_data, batch_size):

        if next(model.parameters()).is_cuda:
            X_train, y_train = X_train.cuda(), y_train.cuda()

        N_data += batch_size
        output = model(X_train)

        if losses is None:
            losses = loss_fn(*output, y_train)
        else:
            losses += loss_fn(*output, y_train)

    return losses, N_data

def make_predictions(data_loader, model, training_data, batch_size):

    assert isinstance(model, nn.Module)

    model.eval()

    y_pred_all = None
    y_test_all = None

    for X_train, y_train in data_loader(training_data, batch_size, random=False):

        is_on_gpu = next(model.parameters()).is_cuda
        if is_on_gpu:
            X_train, y_train = X_train.cuda(), y_train.cuda()

        y_pred = model(X_train)

        if is_on_gpu:
            y_pred, y_train = y_pred.cpu(), y_train.cpu()

        if y_pred_all is None:
            y_pred_all = y_pred.detach().numpy()
            y_test_all = y_train.detach().numpy()

        else:
            y_pred_all = np.concatenate([y_pred_all, y_pred.detach().numpy()])
            y_test_all = np.concatenate([y_test_all, y_train.detach().numpy()])

    model.train()

    return y_pred_all, y_test_all


def extract_features(data_loader, model, training_data, batch_size):

    assert hasattr(model, "return_last_layer"), "The model must have the method return_last_layer implemented"

    model.eval()

    features_all = None
    y_test_all = None

    for X_train, y_train in data_loader(training_data, batch_size, random=False):

        is_on_gpu = next(model.parameters()).is_cuda
        if is_on_gpu:
            X_train, y_train = X_train.cuda(), y_train.cuda()

        features = model.return_last_layer(X_train)

        if is_on_gpu:
            features, y_train = features.cpu(), y_train.cpu()

        if features_all is None:
            features_all = features.detach().numpy()
            y_test_all = y_train.detach().numpy()

        else:
            features_all = np.concatenate([features_all, features.detach().numpy()])
            y_test_all = np.concatenate([y_test_all, y_train.detach().numpy()])

    model.train()

    return features_all, y_test_all

