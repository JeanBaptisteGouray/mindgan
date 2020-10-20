from torchvision import transforms, datasets
import os
import torch
import pickle as pkl
import numpy as np


class EncodedFFHQ(torch.utils.data.Dataset):
    """Encoded Images dataset."""
    def __init__(self, data_path):
        self.data_path = data_path
        self.liste = os.listdir(data_path)

    def __len__(self):
        return len(self.liste)

    def __getitem__(self, idx):
        with open(self.data_path + '/' + self.liste[idx], 'rb') as f:
            latent = pkl.load(f)

        latent = np.asarray(latent, dtype=np.float32)

        return latent
        

