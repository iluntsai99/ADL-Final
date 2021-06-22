from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random
import torch

class myDataset(Dataset):
    def __init__(self, split, data, tokenized_title, tokenized_maintext):
        self.split = split
        self.data = data
        self.tokenized_title = tokenized_title
        self.tokenized_maintext = tokenized_maintext

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.split == "train" or self.split == "public":
            return torch.tensor(self.tokenized_title["input_ids"][index]), torch.tensor(self.tokenized_maintext["input_ids"][index])
        else:
            return torch.tensor(self.tokenized_maintext["input_ids"][index]), self.data[index]["id"]