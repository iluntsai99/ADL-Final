import torch
from torch.utils.data import Dataset 
import random

class Context_Dataset(Dataset):
    def __init__(self, split, data, tokenized_dialogue, tokenized_choices, label):
        self.split = split
        self.data = data
        self.tokenized_dialogue = tokenized_dialogue
        self.tokenized_choices = tokenized_choices
        self.label = label
        self.max_dialogue_len = 350
        self.max_choice_len = 64

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_dialogue_len + 1 + self.max_choice_len + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_dialogue = [101] + self.tokenized_dialogue[idx].ids[-self.max_dialogue_len:] + [102]
        
        related_tokenized_choices = [c['input_ids'][:self.max_choice_len]+[102] for c in self.tokenized_choices[idx]]

        input_ids, attention_mask, token_type_ids = self.padding(tokenized_dialogue, related_tokenized_choices)

        if self.split == "train" or self.split == "eval":
            return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), torch.tensor(self.label[idx])

        # Testing
        else:
            return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids)#, torch.tensor(self.related_contexts_pack[idx])

    def padding(self, question_id, contexts_id):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for context in contexts_id:
            padding_len = self.max_seq_len - len(question_id) - len(context)
            # print(padding_len)
            input_ids.append(question_id + context + [0]*padding_len)
            attention_mask.append([1]*(len(question_id) + len(context)) + [0]*padding_len)
            token_type_ids.append([0]*len(question_id) + [1]*len(context) + [0]*padding_len)
        return input_ids, attention_mask, token_type_ids