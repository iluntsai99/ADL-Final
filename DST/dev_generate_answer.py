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
			possible_slots = [service + '-' + slot for service in dialogue["services"] for slot in services_list[service]]
			status = {possible_slot : "none" for possible_slot in possible_slots}

			for j in range(0, len(dialogue["turns"]), 2):
				for frame in dialogue["turns"][j]["frames"]:
					for slot_name, slot_value in frame["state"]["slot_values"].items():
						status[frame["service"] + '-' + slot_name] = slot_value[0].lower()
			else:
				status_copy = status.copy()
				for k, v in status_copy.items():
					if v == 'none':
						status.pop(k)
				output[dialogue["dialogue_id"]] = status

output_dir = "./data/dev_answer.json"
with open(output_dir, 'w') as f:
	json.dump(output, f, indent=4)
