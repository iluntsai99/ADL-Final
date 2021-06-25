import json
import os

services_list = {}
schema_dir = "./data/schema.json"

with open(schema_dir) as f:
	schema = json.load(f)
	for service in schema:
		slots = []
		for slot in service["slots"]:
			slots.append(slot["name"])
		services_list[service["service_name"]] = slots

data_dir = "./data/dev"
file_names = sorted(os.listdir(data_dir))

output = {}

for file_name in file_names:
	file_dir = os.path.join(data_dir, file_name)

	with open(file_dir) as f:
		
		dialogues = json.load(f)
		for i, dialogue in enumerate(dialogues):
			possible_slots = [slot for service in dialogue["services"] for slot in services_list[service]]
			status = {possible_slot : "none" for possible_slot in possible_slots}

			for j in range(0, len(dialogue["turns"]), 2):
				if j == 0:
					first_system = "SYSTEM: Hello!"
				else:
					first_system = "SYSTEM: " + dialogue["turns"][j-1]["utterance"]

				user = "USER: " + dialogue["turns"][j]["utterance"]
				second_system = "SYSTEM: " + dialogue["turns"][j+1]["utterance"]
				utterance = first_system + ' ' + user + ' ' + second_system

				prev_status = "< previous status >"
				for k in status:
					prev_status += ' ' + k + ' delimiter ' + status[k] + ' seperate'

				for frame in dialogue["turns"][j]["frames"]:
					for slot_name, slot_value in frame["state"]["slot_values"].items():
						status[slot_name] = slot_value[0]

				INPUT = utterance + ' ' + prev_status
				new_status = "< new status >"
				for k in status:
					new_status += ' ' + k + ' delimiter ' + status[k] + ' seperate' 
			else:
				output[dialogue["dialogue_id"]] = new_status.strip()

output_dir = "./data/dev_processed/dev_answer.json"
with open(output_dir, 'w') as f:
	json.dump(output, f, indent=4)
