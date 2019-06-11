import torch
import torch.nn as nn
from importlib import import_module


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.num_gpu = config.num_gpu
        self.device = config.device
        self.losses = []
        self.loss_module = nn.ModuleList()

        loss_function = nn.NLLLoss()
        self.losses.append({'function': loss_function})

        self.loss_module.to(self.device)
        if not config.cpu and config.num_gpu > 1:
            self.loss_module = nn.DataParallel(self.loss_module,
                                               list(range(self.num_gpu)))

    def forward(self, results, label):
        losses = []
        for i, l in enumerate(self.losses):
            if l['function'] is not None:
                loss = l['function'](results, label)
                effective_loss = loss
                losses.append(effective_loss)

        loss_sum = sum(losses)
        if len(self.losses) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum
