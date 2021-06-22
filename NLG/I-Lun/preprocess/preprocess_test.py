from pathlib import Path
import json
import os
from tqdm import tqdm
# '<|user|>', '<|system|>', '<|chitchat|>'

context_list = list()
def preprocess(directory):
	# iterate through all files in train
	for filename in tqdm(os.listdir(directory)):
		context_path = directory/filename
		contexts = json.loads(context_path.read_text())
		# print(contexts[0])
		
		# each file has several dialogues
		for dialogue in range(len(contexts)):
			context_prefix = ""
			for i in range(len(contexts[dialogue]['turns'])):
				if i % 2 == 0: # user turn
					context_prefix += '<|user|>'+contexts[dialogue]['turns'][i]['utterance']
				else:  # system turn
					chitchat_dict = dict()
					chitchat_dict['dialogue_id'] = contexts[dialogue]['dialogue_id']
					chitchat_dict['context'] = context_prefix
					context_list.append(chitchat_dict)
					context_prefix += '<|system|>'+contexts[dialogue]['turns'][i]['utterance']

preprocess(Path("./adl-final-dst-with-chit-chat-seen-domains/data/data/test_seen"))
print(len(context_list), len(context_list[0]))
with open("dataset/test.json", "w") as f:
	json.dump(context_list, f, indent=2)
