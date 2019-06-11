import torch

from config import get_config
from data import MidiData
from op import Operator
from util import Checkpoint


def main(config):
    config.device = torch.device('cuda:{}'.format(config.gpu)
                                 if torch.cuda.is_available() else 'cpu')

    # load data_loader
    check_point = Checkpoint(config)
    operator = Operator(config, check_point)

    if config.is_train:
        midi_data = MidiData(config)
        operator.train(midi_data)
    else:
        operator.test()


if __name__ == "__main__":
    config = get_config()
    main(config)
