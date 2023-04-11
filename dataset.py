import torch
from torch.utils.data import Dataset


class CryptoDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.Tensor(self.data[idx:idx + self.seq_len])
        y = torch.Tensor([self.data[idx+self.seq_len]])
        return x, y
