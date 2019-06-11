import torch.nn as nn


def make_model(args):
    return GRU(args)


class GRU(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()

        self.n_hidden = config.n_hidden
        self.n_layers = config.n_layers
        self.n_dict = config.n_dict

        # define encoder
        self.encoder = nn.Embedding(self.n_dict + 1, self.n_dict + 1)

        # define rnn cell
        self.rnn = nn.GRU(input_size=self.n_dict,
                           hidden_size=self.n_hidden,
                           num_layers=self.n_layers)

        # define decoder
        self.decoder = nn.Linear(in_features=self.n_hidden,
                                 out_features=self.n_dict+1)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x_encoder = self.encoder(x)
        x_encoder = x_encoder.unsqueeze(0)
        x_encoder, x_hidden = self.rnn(x_encoder)
        x_encoder = x_encoder.squeeze(0)
        x_decoder = self.decoder(x_encoder)
        x_pred = self.log_softmax(x_decoder)

        return x_pred, x_hidden

