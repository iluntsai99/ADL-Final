import json
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
import transformers

import utils
from dataset import myDataset
from accelerate import Accelerator
import time
import random
import os

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/summary_experiment_1')

TRAIN = "train"
DEV = "dev"
SPLITS = [TRAIN, DEV]

def main(args):
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    print(len(data[TRAIN]), data[TRAIN][0]['context'])
    print(data[TRAIN][0]['chit-chat'])

    if (args.start_from_last):
        print("load from last...")
        model = T5ForConditionalGeneration.from_pretrained(args.ckpt_dir).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|system|>', '<|chitchat|>']})
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer)) 
    train_chitchat_tokenized = tokenizer([train_data["chit-chat"] for train_data in data[TRAIN]], return_tensors="pt", truncation=True, max_length=args.max_chitchat_len, padding=True)
    dev_chitchat_tokenized = tokenizer([dev_data["chit-chat"] for dev_data in data[DEV]], return_tensors="pt", truncation=True, max_length=args.max_chitchat_len, padding=True)
    train_context_tokenized = tokenizer([train_data["context"] for train_data in data[TRAIN]], return_tensors="pt", truncation=True, max_length=args.max_context_len, padding=True)
    dev_context_tokenized = tokenizer([dev_data["context"] for dev_data in data[DEV]], return_tensors="pt", truncation=True, max_length=args.max_context_len, padding=True)
    # train_chitchat_tokenized = utils.create_labels(train_chitchat_tokenized)
    # dev_chitchat_tokenized = utils.create_labels(dev_chitchat_tokenized)
    # train_context_tokenized = utils.create_labels(train_context_tokenized)
    # dev_context_tokenized = utils.create_labels(dev_context_tokenized)
    # print(train_context_tokenized['input_ids'][0])
    # print(train_context_tokenized['attention_mask'][0])
    # print(train_context_tokenized['labels'][0])
    # print(dev_context_tokenized['input_ids'][0])
    # print(dev_context_tokenized['attention_mask'][0])
    # print(dev_context_tokenized['labels'][0])
    # exit()
    train_set = myDataset(TRAIN, data[TRAIN], train_context_tokenized, train_chitchat_tokenized)
    dev_set = myDataset(DEV, data[DEV], dev_context_tokenized, dev_chitchat_tokenized)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # optimizer = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    update_step = args.num_epoch * len(train_loader) // args.gradient_accumulation_step + args.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * update_step, update_step)

    best_loss = 10000
    for epoch in range(args.num_epoch):
        train_size, dev_size = len(train_loader.dataset), len(dev_set)
        print(train_size, dev_size)
        model.train()
        train_loss = 0
        for i, datas in enumerate(tqdm(train_loader)):
            datas = [data.to(device) for data in datas]
            # print(datas[1].shape, datas[0].shape)
            output = model(input_ids=datas[0], attention_mask=datas[1], labels=datas[2])

            train_loss += output.loss
            normalized_loss = output.loss / args.gradient_accumulation_step
            
            if fp16_training:
                accelerator.backward(normalized_loss)
            else:
                normalized_loss.backward()
            
            if i % args.gradient_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            if i % args.logging_step == 0 and i != 0:
                # Print training loss and accuracy over past logging step
                writer.add_scalar("Loss/train", train_loss.item() / len(train_loader), epoch * len(train_loader))
                print(f"Epoch {epoch + 1}/{args.num_epoch} | loss = {train_loss.item() / len(train_loader):.3f} lr = {optimizer.param_groups[0]['lr']:.6f}")
                dev_loss = 0
                model.eval()
                with torch.no_grad():
                    dev_acc = 0
                    randomlist = random.sample(range(0, len(dev_set)), len(dev_set))
                    dev_subset = Subset(dev_set, randomlist)
                    dev_loader = DataLoader(dev_subset, batch_size=args.batch_size, shuffle=False)
                    for j, datas in enumerate(dev_loader):
                        datas = [data.to(device) for data in datas]
                        output = model(datas[1], labels=datas[1])
                        dev_loss += (output.loss.item() / len(dev_loader))
                        print(f"Validation | Steps {j}/{len(dev_loader)} | loss = {dev_loss:.3f}", end="\r")

                    writer.add_scalar("Loss/dev", dev_loss, epoch * len(train_loader))
                    print(f"Validation | Steps {j}/{len(dev_loader)} | loss = {dev_loss:.3f}")

                    print("Saving model...with loss: {}".format(dev_loss))
                    best_loss = dev_loss
                    model.save_pretrained(args.ckpt_dir/f"model_{dev_loss}")
                    model.train()

    writer.flush()
    writer.close()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/adaf/",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_chitchat_len", type=int, default=64)
    parser.add_argument("--max_context_len", type=int, default=512)

    # training
    parser.add_argument("--start_from_last", action="store_true")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--gradient_accumulation_step", type=int, default=32)
    parser.add_argument("--logging_step", type=int, default=750)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    utils.same_seeds(6396969)
    fp16_training = False
    if fp16_training:
        accelerator = Accelerator(fp16=True)
        device = accelerator.device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)