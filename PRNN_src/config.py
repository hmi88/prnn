import argparse
import os
from distutils.util import strtobool

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--is_train", type=strtobool, default='true')
parser.add_argument("--tensorboard", type=strtobool, default='true')
parser.add_argument("--is_resume", action='store_true', help='resume')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--exp_dir", type=str, default="../PRNN_exp")
parser.add_argument("--exp_load", type=str, default=None)

# Data
parser.add_argument("--data_dir", type=str, default="/mnt/sda")
parser.add_argument("--data_name", type=str, default="epiano")
parser.add_argument("--data_type",  default='event', choices=('event', 'note'))
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--window_size', type=int, default=200)

# Model
parser.add_argument('--cell', default='lstm', choices=('lstm', 'gru'))
parser.add_argument('--n_dict', type=int, default=256)
parser.add_argument('--n_hidden', type=int, default=512)
parser.add_argument('--n_layers', type=int, default=3)

# Train
parser.add_argument("--epochs", type=int, default=4000)
parser.add_argument("--decay", type=str, default='1000-1000-1000-1000')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--epsilon", type=float, default=1e-8)

# Test
parser.add_argument("--sequence", type=int, default=1000)
parser.add_argument("--init_tempo", type=int, default=130)


def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    return config
