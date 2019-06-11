import numpy as np
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from model import *
from loss import Loss
from util import make_optimizer
import data

class Operator:
    def __init__(self, config, ckeck_point):
        self.ckpt = ckeck_point
        self.config = config
        self.epochs = config.epochs
        self.device = config.device
        self.sequence = config.sequence
        self.data_type = config.data_type
        self.tensorboard = config.tensorboard
        if self.tensorboard:
            self.summary_writer = SummaryWriter(self.ckpt.log_dir, 300)

        # set model, criterion, optimizer
        self.model = Model(config)
        self.criterion = Loss(config)
        self.optimizer = make_optimizer(config, self.model)

        # load ckpt, model, optimizer
        if config.is_resume or not config.is_train:
            print("Loading model... ")
            self.load(self.ckpt)
            print(self.ckpt.last_epoch, self.ckpt.global_step)

    def train(self, midi_data):
        last_epoch = self.ckpt.last_epoch
        data_loader = midi_data.get_dataloader()
        train_batch_num = len(data_loader['train'])
        self.model.train()

        for epoch in range(last_epoch, self.epochs):
            for batch_idx, batch_data in enumerate(data_loader['train']):
                batch_data = batch_data.to(self.device)
                batch_c0, batch_h0 = self.model.init_hidden(batch_data.shape[0])
                init_hidden = (batch_c0.to(self.device), batch_h0.to(self.device))
                batch_hidden = init_hidden
                loss = 0.0

                # forward
                for step in range(batch_data.shape[1] - 1):
                    pred, hidden = self.model(x=batch_data[:, step],
                                              hidden=batch_hidden)
                    loss += self.criterion(pred, batch_data[:, step+1])

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:5f}'
                      .format(epoch, self.config.epochs, batch_idx,
                              train_batch_num, loss.item()))

                # use tensorboard
                if self.tensorboard:
                    current_global_step = self.ckpt.step()
                    self.summary_writer.add_scalar('train/loss',
                                                   loss, current_global_step)

            # use tensorboard
            if self.tensorboard:
                print(self.optimizer.get_lr(), epoch)
                self.summary_writer.add_scalar('epoch_lr',
                                               self.optimizer.get_lr(), epoch)

            # test model & save model
            self.optimizer.schedule()
            if (epoch % 50) == 1:
                self.save(self.ckpt, epoch)
                self.test()
                self.model.train()

        self.summary_writer.close()

    def test(self):
        with torch.no_grad():
            self.model.eval()
            now = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            save_path = os.path.join(self.ckpt.save_dir,
                                     '{}_{}.mid'.format(self.data_type, now))

            batch_c0, batch_h0 = self.model.init_hidden(batch_size=1,
                                                        random_init=False)
            init_hidden = (batch_c0.to(self.device), batch_h0.to(self.device))
            init_pred = torch.zeros((1,), dtype=torch.long).to(self.device)

            hidden = init_hidden
            pred = init_pred
            preds = []

            for step in range(self.sequence - 1):
                pred, hidden = self.model(x=pred, hidden=hidden)
                pred_dist = pred.data.view(-1).exp()
                pred = torch.multinomial(pred_dist, 1)
                preds.append(pred.cpu().numpy()[0])

            if self.data_type == 'event':
                data.MidiData.Event2Midi(preds, save_path)
            # elif self.data_type == 'note':
            #     midi_data.Note2Midi(preds, save_path)

    def load(self, ckpt):
        ckpt.load() # load ckpt
        self.model.load(ckpt) # load model
        self.optimizer.load(ckpt) # load optimizer

    def save(self, ckpt, epoch):
        ckpt.save(epoch) # save ckpt: global_step, last_epoch
        self.model.save(ckpt, epoch) # save model: weight
        self.optimizer.save(ckpt) # save optimizer:



