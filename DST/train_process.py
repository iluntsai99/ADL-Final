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

data_dir = "./data/train"
file_names = sorted(os.listdir(data_dir))

output_dir = "./data/train_processed.jsonl"
output_file = open(output_dir, 'w')

for file_name in file_names:
	file_dir = os.path.join(data_dir, file_name)

	with open(file_dir) as f:
		
		dialogues = json.load(f)
		for i, dialogue in enumerate(dialogues):
			
			# Concat all possible slots into a list
			possible_slots = [slot for service in dialogue["services"] for slot in services_list[service]]

			# Initial status is empty
			status = {possible_slot : "<extra_id_6>" for possible_slot in possible_slots}

			##### Format of each training data #####
			# SYSTEM (SYSTEM utterance) USER (USER utterance) SYSTEM (SYSTEM utterance) USER (USER utterance) SYSTEM (SYSTEM utterance)
			# < previous status > slot1 : value1 , slot2 : value2 , ... , slotn, valuen
			# < new status > (generate here)
			# Note1: Overlap 2 utterances
			# Note2: For first / second utterance, fake SYSTEM / USER utterance is added
			# Note3: Every states["slot_values"] is put in USER utterance in training data
			# Note4: Output answer is formed by combining every status (the latter always has higher priority)

			# Special Tokens Mapping
			# SYSTEM : <extra_id_0>
			# USER   : <extra_id_1>
			# start of previous status : <extra_id_2>
			# start of new status : <extra_id_3>
			# slot_name <extra_id_4> slot_value <extra_id_5>
			# empty state : <extra_id_6>

			for j in range(0, len(dialogue["turns"]), 2):
				if j == 0:
					first_system = "<extra_id_0>hi"
					first_user = "<extra_id_1>hi"
					second_system = "<extra_id_0>hi"
				elif j == 2:
					first_system = "<extra_id_0>hi"
					first_user = "<extra_id_1>" + dialogue["turns"][j-2]["utterance"]
					second_system = "<extra_id_0>" + dialogue["turns"][j-1]["utterance"]
				else:
					first_system = "<extra_id_0>" + dialogue["turns"][j-3]["utterance"]
					first_user = "<extra_id_1>" + dialogue["turns"][j-2]["utterance"]
					second_system = "<extra_id_0>" + dialogue["turns"][j-1]["utterance"]

				second_user = "<extra_id_1>" + dialogue["turns"][j]["utterance"]
				third_system = "<extra_id_0>" + dialogue["turns"][j+1]["utterance"]

				utterance = first_system + first_user + second_system + second_user + third_system

				prev_status = "<extra_id_2>"
				for k in status:
					prev_status += k + '<extra_id_4>' + status[k] + '<extra_id_5>'
				
				for frame in dialogue["turns"][j]["frames"]:
					for slot_name, slot_value in frame["state"]["slot_values"].items():
						status[slot_name] = slot_value[0]

				INPUT = utterance + prev_status
				
				new_status = "<extra_id_3>"
				for k in status:
					new_status += k + '<extra_id_4>' + status[k] + '<extra_id_5>' 

				json.dump({"IN" : INPUT, "OUT" : new_status}, output_file)
				output_file.write('\n')

output_file.close()