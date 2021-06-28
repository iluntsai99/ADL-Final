import json
import os

data_dir = "./data/dev"
file_names = sorted(os.listdir(data_dir))

output_dir = "./data/dev.json"
output_file = open(output_dir, 'w')

test_format_dialogues = []

for file_name in file_names:
	file_dir = os.path.join(data_dir, file_name)
	with open(file_dir) as f:
		dialogues = json.load(f)
		for i, dialogue in enumerate(dialogues):
			test_format_dialogue = {}
			test_format_dialogue["dialogue_id"] = dialogue["dialogue_id"]
			test_format_dialogue["services"] = dialogue["services"]
			test_format_dialogue["turns"] = []
			for j, turn in enumerate(dialogue["turns"]):
				test_format_dialogue["turns"].append({"turn_id" : j, "speaker" : turn["speaker"], "utterance" : turn["utterance"]})
			test_format_dialogues.append(test_format_dialogue)

with open(output_dir, 'w') as f:
	json.dump(test_format_dialogues, output_file, indent=1)

# Transform dev set into test_set format
# test_set format: 	[{dialogue},{dialogue} ...]
#					dialogue:
#						"dialogue_id" 	: str
#						"services" 		: list of str
#						"turns"			: list of {"turn_id" : int, "speaker" : str, "utterance : str"}