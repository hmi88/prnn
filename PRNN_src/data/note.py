import os
import random

import torch
from torch.utils.data import Dataset

import data
from util import find_files_by_extensions


class NoteDataset(Dataset):
    def __init__(self, config):
        super(NoteDataset, self).__init__()
        self.window_size = config.window_size
        self.data_path = os.path.join(config.data_dir,
                                      config.data_name,
                                      config.data_type)
        if not os.path.exists(self.data_path):
            print("Prepare processed data")
            data.MidiData.Midi2Note(config)
        self.data_list = list(find_files_by_extensions(self.data_path, ['.data']))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_path,
                                 self.data_list[idx])
        data = torch.load(file_name)
        random_int = random.randint(0, len(data) - self.window_size)
        data = data[random_int:random_int+self.window_size]
        return data
