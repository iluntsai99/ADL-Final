import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm.auto import tqdm

from transformers import AdamW, BertTokenizerFast, BertForMultipleChoice
from transformers.optimization import get_linear_schedule_with_warmup
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
    train_path = args.data_dir/f"train.json"
    eval_path = args.data_dir/f"dev.json"
    train_data = json.loads(train_path.read_text())
    eval_data = json.loads(eval_path.read_text())

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    train_dialogues = [train_dialogue["context"].split("=")[0] for train_dialogue in train_data]
    train_labels = [train_dialogue["context"].split("=")[1].replace('<|endoftext|>','') for train_dialogue in train_data]
    train_chitchats = [train_dialogue["chit-chat"][0] for train_dialogue in train_data]
    train_choices = [[l.replace('<|chitchat|>'+c,'')+'<|chitchat|>'+c,'<|chitchat|>'+c+l.replace('<|chitchat|>'+c,''),l.replace('<|chitchat|>'+c,'')] for l,c in zip(train_labels,train_chitchats)]
    train_label_indice = [ch.index(l) for l,ch in zip(train_labels, train_choices)]

    eval_dialogues = [eval_dialogue["context"].split("=")[0] for eval_dialogue in eval_data]
    eval_labels = [eval_dialogue["context"].split("=")[1].replace('<|endoftext|>','') for eval_dialogue in eval_data]
    eval_chitchats = [eval_dialogue["chit-chat"][0] for eval_dialogue in eval_data]
    eval_choices = [[l.replace('<|chitchat|>'+c,'')+'<|chitchat|>'+c,'<|chitchat|>'+c+l.replace('<|chitchat|>'+c,''),l.replace('<|chitchat|>'+c,'')] for l,c in zip(eval_labels,eval_chitchats)]
    eval_label_indice = [ch.index(l) for l,ch in zip(eval_labels, eval_choices)]

    train_dialogues_tokenized = tokenizer(train_dialogues, add_special_tokens=False)
    train_labels_tokenized = tokenizer(train_labels, add_special_tokens=False)
    train_choices_tokenized = [[tokenizer(c, add_special_tokens=False) for c in train_choice] for train_choice in train_choices]

    eval_dialogues_tokenized = tokenizer(eval_dialogues, add_special_tokens=False)
    eval_labels_tokenized = tokenizer(eval_labels, add_special_tokens=False)
    eval_choices_tokenized = [[tokenizer(c, add_special_tokens=False) for c in eval_choice] for eval_choice in eval_choices]
    
    
    train_dataset = Context_Dataset('train', train_data, train_dialogues_tokenized, train_choices_tokenized, train_label_indice)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataset = Context_Dataset('eval', eval_data, eval_dialogues_tokenized, eval_choices_tokenized, eval_label_indice)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    device = "cuda"
    model = BertForMultipleChoice.from_pretrained("bert-base-cased") #hfl/chinese-macbert-base

    model = model.to(device)

    num_epoch = 2 #2 or 3
    validation = True
    logging_step = 1000
    learning_rate = 5e-5 # 3e-5
    pretrain_step = 4000
    validate_step = 1000

    # gradient accumulation
    accum_iter = 32 # 8 or 16
    # validate more time

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    update_step = num_epoch*len(train_loader) // accum_iter
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*update_step, num_training_steps=update_step)

    
    best_acc = 0
    step = 1
    for epoch in range(num_epoch):
        train_loss = train_acc = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):	
            data = [i.to(device) for i in data] # input_ids, attention_mask, token_type_ids, label
            # print(data)
            output = model(input_ids=data[0], attention_mask=data[1], token_type_ids=data[2], labels=data[3])

            train_acc += (torch.argmax(output.logits, dim=1)==data[3]).float().mean()
            loss = output.loss
            train_loss += loss.mean().item()
            # loss = output.loss/accum_iter
            loss.mean().backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            step += 1

            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                # print(optimizer.param_groups[0]['lr'])
                train_loss = train_acc = 0

            if validation:
                if step > pretrain_step and (step % validate_step==0):
                    print("Evaluating Eval Set...")
                    model.eval()
                    with torch.no_grad():
                        eval_acc = 0
                        for idx, data in enumerate(eval_loader):
                            print(idx+1,"/",len(eval_loader),end='\r')
                            data = [i.to(device) for i in data] # input_ids, attention_mask, token_type_ids, label
                            output = model(input_ids=data[0], attention_mask=data[1], token_type_ids=data[2], labels=data[3])
                            eval_acc += (torch.argmax(output.logits, dim=1)==data[3]).float().mean()
                        print(f"Validation | Epoch {epoch + 1} | acc = {eval_acc / len(eval_loader):.3f}")
                    if eval_acc > best_acc:
                        best_acc = eval_acc
                        print("Saving Model ...")
                        model_save_dir = "filter_model_base" 
                        
                        torch.save(model, model_save_dir)
                        print("Model saved!")
                    model.train()
    
    




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)