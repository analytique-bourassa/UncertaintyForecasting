import torch

import numpy as np
from tqdm import tqdm
from torch.distributions.normal import Normal

from models.LSTM_Bayesian.LSTM import LSTM
from models.LSTM_Bayesian.training_parameters import TrainingParameters
from models.monitoring.training_monitoring import LossesMonitor

KEY_VALIDATION = "validation"

class CheckpointSaver(object):

    def __init__(self, path, name):

        self._path = path
        self._name = name

    def __call__(self, model, optimizer, suffix):

        torch.save(model.state_dict(),
                   self._path + "model_" + self._name + "_" + suffix)

        torch.save(optimizer.state_dict(),
                   self._path + "optimizer_" + self._name + "_" + suffix)


def load_checkpoint(model, optimizer, path, name):
    model.load_state_dict(torch.load(path + "model_" + name), strict=False)
    optimizer.load_state_dict(torch.load(path + "optimizer_" + name))


def train_model(model, optimizer, dataset, loss_fn, training_parameters):

    assert isinstance(model, LSTM)
    assert isinstance(training_parameters, TrainingParameters)
    assert training_parameters.batch_size is not None, "batch size must be defined in training parameters"

    if training_parameters.do_monitor_training_loss:
        losses_monitor = LossesMonitor()

    save_checkpoint = CheckpointSaver(path=training_parameters.checkpoints_saving_path,
                                      name=training_parameters.checkpoints_saving_name)

    for epoch in tqdm(range(1, training_parameters.num_epochs + 1)):

        try:
            model.hidden = model.init_hidden()
            losses, n_data_train = make_forward_pass(dataset, model, training_parameters, loss_fn)
            train_loss = losses.item() / n_data_train
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if training_parameters.do_monitor_training_loss:
                losses_monitor.append_values(epoch, train_loss, set="train")

            if epoch % training_parameters.checkpoints_intervals == 0:

                if training_parameters.save_checkpoints_bool:
                    save_checkpoint(model=model,
                                    optimizer=optimizer,
                                    suffix="_epoch_%d" % epoch)

                model.eval()
                losses, n_data_val = make_forward_pass(dataset, model, training_parameters, loss_fn, set=KEY_VALIDATION)
                val_loss = losses.item()/n_data_val
                model.train()

                print("Epoch %d | MSE (train): %0.5f | MSE (val) %0.5f" % (epoch, train_loss, val_loss))

                if training_parameters.do_monitor_training_loss:
                    losses_monitor.append_values(epoch, val_loss, set="val")

        except KeyboardInterrupt:
            print("Keyboard interrupt: saving checkpoint...")
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            suffix="_epoch_%d" % epoch)
            print("Checkpoint saved.")
            break

    if training_parameters.save_checkpoints_bool:
        save_checkpoint(model=model,
                        optimizer=optimizer,
                        suffix="_end")

    if training_parameters.do_monitor_training_loss:
        return losses_monitor


def is_loss_ratio_below_tolerance(previous_loss, losses, tolerance):

    ratio_change_of_loss = abs((previous_loss - losses) / previous_loss)
    return ratio_change_of_loss < tolerance


def is_loss_above_tolerance(previous_min_loss, losses, tolerance):

    ratio_change_of_loss = (losses - previous_min_loss) / previous_min_loss
    return ratio_change_of_loss > tolerance


def make_predictions(model,
                     dataset,
                     training_parameters,
                     set="train",
                     add_noise=False,
                     noise_level=1.0,
                     feature_index_noise=0,
                     feature_to_drop=None):
    """
       NOTE: the feature index for the noise is based on the keeped indexes!!
    """

    #assert set in ["train", KEY_VALIDATION]
    model.eval()

    y_pred_all = None
    y_test_all = None

    for X_train, y_train in dataset.data_loader(training_parameters.batch_size, set=set, random=False):

        if add_noise:

            shape_for_noise = X_train[:, :, feature_index_noise].size()
            noise_generator = Normal(loc=0, scale=noise_level)
            X_train[:, :, feature_index_noise] += noise_generator.rsample(shape_for_noise)

        if feature_to_drop is not None:
            shape_for_zeros = X_train[:, :, feature_to_drop].size()
            X_train[:, :, feature_to_drop] = torch.zeros(shape_for_zeros)

        if training_parameters.use_gpu:
            X_train, y_train = X_train.cuda(), y_train.cuda()

        y_pred = model(X_train)

        if training_parameters.use_gpu:
            y_pred = y_pred.cpu()
            y_train = y_train.cpu()

        if y_pred_all is None:
            y_pred_all = y_pred.detach().numpy()
            y_test_all = y_train.detach().numpy()

        y_pred_all = np.concatenate([y_pred_all, y_pred.detach().numpy()])
        y_test_all = np.concatenate([y_test_all, y_train.detach().numpy()])

    model.train()

    return y_pred_all, y_test_all


def make_forward_pass(dataset, model, training_parameters, loss_fn, set="train"):

    losses = None
    N_data = 0

    for X_train, y_train in dataset.data_loader(training_parameters.batch_size, set=set, random=True):

        N_data += training_parameters.batch_size
        if training_parameters.use_gpu:
            X_train = X_train.cuda()
            y_train = y_train.cuda()

        y_pred = model(X_train)

        if losses is None:
            losses = loss_fn(y_pred, y_train)
        else:
            losses += loss_fn(y_pred, y_train)

    return losses, N_data
