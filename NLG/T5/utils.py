import torch
import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_labels(inputs):
    labels=[]
    for ids, attention_mask in zip(inputs['input_ids'], inputs['attention_mask']):
        label = ids.clone()
        real_len = sum(attention_mask)
        padding_len = len(attention_mask) - sum(attention_mask)
        label[:] = torch.cat((label[:real_len], torch.tensor([-100] * padding_len)), dim=-1)
        labels.append(label)
    inputs['labels'] = labels
    return inputs