import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        print('Making model...')

        self.device = config.device
        self.is_train = config.is_train
        self.num_gpu = config.num_gpu
        self.cell = config.cell
        self.n_layers = config.n_layers
        self.n_hidden = config.n_hidden

        module = import_module('model.' + self.cell)
        self.model = module.make_model(config).to(self.device)

    def forward(self, x, hidden):
        if self.num_gpu > 1:
            return P.data_parallel(self.model, (x, hidden),
                                   list(range(self.num_gpu)))
        else:
            return self.model(x, hidden)

    def init_hidden(self, batch_size, random_init=True):
        if random_init:
            return torch.randn(self.n_layers, batch_size, self.n_hidden), \
                   torch.randn(self.n_layers, batch_size, self.n_hidden)
        else:
            return torch.zeros(self.n_layers, batch_size, self.n_hidden), \
                   torch.zeros(self.n_layers, batch_size, self.n_hidden)

    def save(self, ckpt, epoch):
        save_dirs = [os.path.join(ckpt.model_dir, 'model_latest.pt')]
        save_dirs.append(
            os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, ckpt, cpu=False):
        epoch = ckpt.last_epoch
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if epoch == -1:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_latest.pt'), **kwargs)
        else:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
