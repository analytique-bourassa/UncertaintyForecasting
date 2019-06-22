import torch

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

def save_checkpoint(model, optimizer, path, name):

    torch.save(model.state_dict(),
               path + "model_" + name)

    torch.save(optimizer.state_dict(),
               path + "optimizer_" + name)
