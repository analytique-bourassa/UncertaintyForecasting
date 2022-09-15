import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, saving_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.best_score = None
        self.best_model = None

        self.early_stop = False
        self.val_loss_min = np.Inf


    def __call__(self, epoch, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = model

        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_checkpoint(epoch, val_loss, self.best_model)
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self,epoch,  val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        #torch.save(model.state_dict(), 'checkpoint_%d.pt' % epoch)
        self.val_loss_min = val_loss

