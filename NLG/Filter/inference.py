import json
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm

from transformers import BertTokenizerFast, BertForMultipleChoice, BertForQuestionAnswering
from dataset import Context_Dataset
import torch
from torch.utils.data import DataLoader
import random
import numpy as np

def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)
		    torch.cuda.manual_seed_all(seed)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True

def main(args):
    same_seeds(0)
    test_path = args.data_dir/f"result.json"
    dialogue_path = args.data_dir/f"test.json"

    test_data = json.loads(test_path.read_text())
    dialogue_data = json.loads(dialogue_path.read_text())

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    test_dialogue = []
    test_choices = []

    count = 0
    for key, value in test_data.items():
        for k, v in value.items():
            test_dialogue.append(dialogue_data[count]['context'].split('=')[0])
            chitchat_str_1 = '<|chitchat|>'+v['start']
            chitchat_str_2 = '<|chitchat|>'+v['end']
            test_choices.append([chitchat_str_1+dialogue_data[count]['system'],dialogue_data[count]['system']+chitchat_str_2,dialogue_data[count]['system']])
            count += 1


    test_dialogues_tokenized = tokenizer(test_dialogue, add_special_tokens=False)
    test_choices_tokenized = [[tokenizer(c, add_special_tokens=False) for c in test_choice] for test_choice in test_choices]

    test_dataset = Context_Dataset('test', test_dialogue, test_dialogues_tokenized, test_choices_tokenized, [])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = "cuda"
    print("Loading model...")
    model = BertForMultipleChoice.from_pretrained("bert-base-cased")
    model = torch.load('filter_model_base').to(device) #macbert-base
    print("Finished loading!")

    result = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader)):
            data = [i.to(device) for i in data] # input_ids, attention_mask, token_type_ids
            output = model(input_ids=data[0], attention_mask=data[1], token_type_ids=data[2])
            result.append(torch.argmax(output.logits, dim=1))

    print(len(result))
    count = 0
    for key, value in test_data.items():
        for k, v in value.items():
            if result[count] == 0:
                v['end'] = ''
            elif result[count] == 1:
                v['start'] = ''
            else:
                v['start'] = ''
                v['end'] = ''
            count += 1

    result_file = args.output_file
    print("Saving results to", result_file)
    with open(result_file, 'w', encoding='utf-8') as f:	
        json.dump(test_data, f, ensure_ascii=False, indent=2)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Path to the output file",
        default="./output.json",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)