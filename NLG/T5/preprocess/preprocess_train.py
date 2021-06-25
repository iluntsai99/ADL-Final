from pathlib import Path
import json
import os
from tqdm import tqdm

context_list = list()
def preprocess(directory):
	# iterate through all files in train
	for filename in tqdm(os.listdir(directory)):
		context_path = directory/filename
		contexts = json.loads(context_path.read_text())
		# print(contexts[0])
		# print(filename)
		
		# each file has several dialogues
		for dialogue in range(len(contexts)):
			context_prefix = ""
			for i in range(len(contexts[dialogue]['turns'])):
				if i % 2 == 0: # user turn
					context_prefix += '<|user|>'+contexts[dialogue]['turns'][i]['utterance']
				else:  # system turn
					chitchat_dict = dict()
					chitchat_dict['dialogue_id'] = contexts[dialogue]['dialogue_id']
					chitchat_dict['context'] = ""
					begin, end = False, False
					if 'beginning' in contexts[dialogue]['turns'][i] and contexts[dialogue]['turns'][i]['beginning']:
						new_chitchat = ""
						for j in range(len(contexts[dialogue]['turns'][i]['beginning'])):
							if contexts[dialogue]['turns'][i]['beginning'][j]['label'] == "good":
								new_chitchat += contexts[dialogue]['turns'][i]['beginning'][j]['candidate']
						if new_chitchat != "":
							chitchat_dict['context'] = context_prefix+'=<|chitchat|>'+new_chitchat+'<|system|>'+contexts[dialogue]['turns'][i]['utterance']+'</s>'
							begin = True
					if 'end' in contexts[dialogue]['turns'][i] and contexts[dialogue]['turns'][i]['end']:
						# print("list", len(contexts[dialogue]['turns'][i]['end']))
						new_chitchat = ""
						for j in range(len(contexts[dialogue]['turns'][i]['end'])):
							if contexts[dialogue]['turns'][i]['end'][j]['label'] == "good":
								# print(i, j, contexts[dialogue]['turns'][i]['end'][j]['candidate'])
								new_chitchat += contexts[dialogue]['turns'][i]['end'][j]['candidate']
						if new_chitchat != "":
							if begin:
								chitchat_dict['context'] = chitchat_dict['context'].replace('</s>', '')
								chitchat_dict['context'] += '<|chitchat|>'+new_chitchat+'</s>'
							else:
								chitchat_dict['context'] = context_prefix+'=<|system|>'+contexts[dialogue]['turns'][i]['utterance']+'<|chitchat|>'+new_chitchat+'</s>'
							end = True	
					if begin or end:
						context_prefix += chitchat_dict['context'].replace(context_prefix, '').replace('=', '').replace('</s>', '')
						try:
							context, chitchat = chitchat_dict['context'].split('=')
						except:
							context, chitchat_1, chitchat_2 = chitchat_dict['context'].split('=')
							chitchat = chitchat_1 + chitchat_2
						chitchat_dict['context'] = context+'<|blank|>'+contexts[dialogue]['turns'][i]['utterance']+'<|blank|>'
						chitchat_dict['chit-chat'] = chitchat
						context_list.append(chitchat_dict)
					else:
						context_prefix += '<|system|>'+contexts[dialogue]['turns'][i]['utterance']

preprocess(Path("./adl-final-dst-with-chit-chat-seen-domains/data-0614/data-0614/train"))
print(len(context_list), len(context_list[0]))
with open("dataset/train.json", "w") as f:
	json.dump(context_list, f, indent=2)

context_list = list()
preprocess(Path("./adl-final-dst-with-chit-chat-seen-domains/data-0614/data-0614/dev"))
print(len(context_list), len(context_list[0]))
with open("dataset/dev.json", "w") as f:
	json.dump(context_list, f, indent=2)
