from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random
import torch

class myDataset(Dataset):
    def __init__(self, split, data, tokenized_context):
        self.split = split
        self.data = data
        self.tokenized_context = tokenized_context

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.split == "train" or self.split == "dev":
            return torch.tensor(self.tokenized_context["input_ids"][index]), \
                    torch.tensor(self.tokenized_context["attention_mask"][index]), \
                    torch.tensor(self.tokenized_context["labels"][index])
        else:
            return torch.tensor(self.tokenized_context["input_ids"][index]), \
                    torch.tensor(self.tokenized_context["attention_mask"][index]), \
                    torch.tensor(self.tokenized_context["labels"][index]), self.data[index]["dialogue_id"]