import json
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import Adafactor
import transformers

import utils
from dataset import myDataset
from accelerate import Accelerator
import time
import random
import os

TEST = "test"

def main(args):
    SPLITS = [TEST]
    data_paths = {split: args.data_path for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    print(len(data[TEST]))

    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_dir).to(device)
    tokenizer = T5TokenizerFast.from_pretrained('t5-large')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|user|>', '<|system|>', '<|chitchat|>']})
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer)) 
    test_context_tokenized = tokenizer([test_data["context"] for test_data in data[TEST]], return_tensors="pt", truncation=True, max_length=args.max_context_len, padding=True)
    test_set = myDataset(TEST, data[TEST], test_context_tokenized, None)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model.eval()
    results = dict()
    prev_id = ""
    dialogue, turn =0, 1
    with torch.no_grad():
        for i, datas in enumerate(tqdm(test_loader)):
            outputs = model.generate(input_ids=datas[0].to(device), attention_mask=datas[1].to(device), max_length=args.max_chitchat_len, num_beams=10, repetition_penalty=2.5, do_sample=True, use_cache=True)
            # print(outputs.shape)
            for j in range(outputs.shape[0]):
                gen = tokenizer.decode(outputs[j], skip_special_tokens=False)
                print("generated:", gen)
                try:
                    start = gen.index('<|chitchat|>')+12
                    end = gen.index('<', start)
                    print("=========chit-chat time!=========")
                except:
                    start, end = 0, 0
                print("start:{}, end:{}".format(start, end))
                print(gen[start:end])
                cur_id = datas[2][j]
                if prev_id != cur_id:
                    if prev_id != "":
                        results[prev_id] = dialogue
                        # print(results)
                    dialogue = dict()
                    prev_id = cur_id
                    turn = 1
                dialogue[str(turn)] = {'start': gen[start:end], 'end': gen[start:end], 'mod': ""}
                turn += 2
    # for last dialogue
    results[prev_id] = dialogue

    result_file = args.pred_path
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)	
    print(f"Completed! Result is in {result_file}")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset/test.json",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/model/",
    )
    parser.add_argument(
        "--pred_path",
        type=Path,
        help="Prediction file",
        default="./result.json",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_chitchat_len", type=int, default=64)
    parser.add_argument("--max_context_len", type=int, default=512)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    utils.same_seeds(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    main(args)
