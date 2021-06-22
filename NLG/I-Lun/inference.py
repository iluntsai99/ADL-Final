import json
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from transformers import MT5ForConditionalGeneration, T5TokenizerFast
from transformers import Adafactor
import transformers

import utils
from dataset import myDataset
from accelerate import Accelerator
import time
import random
import os

TEST = "public"

def main(args):
    SPLITS = [TEST]
    data_paths = {split: args.data_path for split in SPLITS}
    data = {split: [json.loads(jline) for jline in path.read_text().splitlines()] for split, path in data_paths.items()}
    # print(len(data[TRAIN]), data[TRAIN][0]['title'])
    # print(data[TRAIN][0]['maintext'])

    model = MT5ForConditionalGeneration.from_pretrained(args.ckpt_dir).to(device)
    tokenizer = T5TokenizerFast.from_pretrained("google/mt5-small")
    with tokenizer.as_target_tokenizer():
        test_maintext_tokenized = tokenizer([test_data["maintext"] for test_data in data[TEST]], return_tensors="pt", truncation=True, max_length=args.max_maintext_len, padding=True)
    test_set = myDataset("TEST", data[TEST], None, test_maintext_tokenized)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model.eval()
    results = list()
    with torch.no_grad():
        for i, datas in enumerate(tqdm(test_loader)):
            outputs = model.generate(input_ids=datas[0].to(device), max_length=args.max_title_len, num_beams=10, repetition_penalty=2.5, do_sample=True, use_cache=True)
            for j in range(outputs.shape[0]):
                summary = dict()
                summary["title"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)[j]
                summary["id"] = datas[1][j]
                # print(summary)
                results.append(summary)

    result_file = args.pred_path
    with open(result_file, 'w', encoding='utf-8') as f:	
        for line in results:
            json_result = json.dumps(line, ensure_ascii=False)
            f.write(json_result + '\n')
    print(f"Completed! Result is in {result_file}")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Directory to the dataset.",
        default="./ADL21-HW3/data/public.jsonl",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/model",
    )
    parser.add_argument(
        "--pred_path",
        type=Path,
        help="Prediction file",
        default="./results.jsonl",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_title_len", type=int, default=64)
    parser.add_argument("--max_maintext_len", type=int, default=512)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    utils.same_seeds(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    main(args)
