import json
import os
import sys

turn, split, evaluate_model_output_dir = int(sys.argv[1]), sys.argv[2], sys.argv[3]

if split == "dev":
	data_dir = "./data"
	filenames = ["dev.json"]
	run_summarization_input_file = "dev_turns.jsonl"
	run_summarization_output_file = os.path.join(evaluate_model_output_dir, "generated_predictions.txt")
	all_status_file = "dev_status.json"
	
elif split == "test_seen":
	data_dir = "./data/test_seen"
	filenames = sorted(os.listdir(data_dir))
	run_summarization_input_file = "test_turns.jsonl"
	run_summarization_output_file = os.path.join(evaluate_model_output_dir, "generated_predictions.txt")
	all_status_file = "test_status_seen.json"
	
elif split == "test_unseen":
	data_dir = "./data/test_unseen"
	filenames = sorted(os.listdir(data_dir))
	run_summarization_input_file = "test_turns_unseen.jsonl"
	run_summarization_output_file = os.path.join(evaluate_model_output_dir, "generated_predictions.txt")
	all_status_file = "test_status_unseen.json"
	

input_file = open(run_summarization_input_file, 'w')

services_list = {}
schema_dir = "./data/schema.json"

with open(schema_dir) as f:
	schema = json.load(f)
	for service in schema:
		slots = []
		for slot in service["slots"]:
			slots.append(slot["name"])
		services_list[service["service_name"]] = slots

if turn == 0:
	all_status = {}
	ID = 0

	for file_name in filenames:
		file_dir = os.path.join(data_dir, file_name)
		
		with open(file_dir, 'r') as f:
			data = json.load(f)
			for i, dialogue in enumerate(data):

				possible_slots = [slot for service in dialogue["services"] for slot in services_list[service]]
				status = {possible_slot : "<extra_id_6>" for possible_slot in possible_slots}

				first_system = "<extra_id_0>hi"
				first_user = "<extra_id_1>hi"
				second_system = "<extra_id_0>hi"
				second_user = "<extra_id_1>" + dialogue["turns"][0]["utterance"]
				third_system = "<extra_id_0>" + dialogue["turns"][1]["utterance"]
				utterance = first_system + first_user + second_system + second_user + third_system

				prev_status = ''
				for slot in possible_slots:
					prev_status += slot + '<extra_id_4>' + "<extra_id_6>" + '<extra_id_5>'
				INPUT = utterance + '<extra_id_2>' + prev_status

				all_status[dialogue["dialogue_id"]] = {"ID" : ID, "status" : prev_status}
				
				json.dump({"IN" : INPUT, "OUT" : ""}, input_file)
				input_file.write('\n')

				ID += 1

	with open(all_status_file, 'w') as f:
		json.dump(all_status, f, indent=4)

else:
	output_file = open(run_summarization_output_file, 'r')
	output_data = output_file.read().splitlines()

	with open(all_status_file, 'r') as f:
		all_status = json.load(f)

	ID = 0

	for file_name in filenames:
		file_dir = os.path.join(data_dir, file_name)
		
		with open(file_dir, 'r') as f:
			data = json.load(f)
			for i, dialogue in enumerate(data):
				if len(dialogue["turns"]) < (turn + 1) * 2:
					continue

				if turn == 1:
					first_system = "<extra_id_0>hi"
					first_user = "<extra_id_1>" + dialogue["turns"][turn * 2 - 2]["utterance"]
					second_system = "<extra_id_0>" + dialogue["turns"][turn * 2 -1]["utterance"]
				else:
					first_system = "<extra_id_0>" + dialogue["turns"][turn * 2 - 3]["utterance"]
					first_user = "<extra_id_1>" + dialogue["turns"][turn * 2 - 2]["utterance"]
					second_system = "<extra_id_0>" + dialogue["turns"][turn * 2 - 1]["utterance"]
				
				second_user = "<extra_id_1>" + dialogue["turns"][turn * 2]["utterance"]
				third_system = "<extra_id_0>" + dialogue["turns"][turn * 2 + 1]["utterance"]
				utterance = first_system + first_user + second_system + second_user + third_system

				prev_status = output_data[all_status[dialogue["dialogue_id"]]["ID"]]
				prev_status = prev_status.replace('<pad>', '').replace('</s>', '')
				INPUT = utterance + '<extra_id_2>' + prev_status

				all_status[dialogue["dialogue_id"]] = {"ID" : ID, "status" : prev_status}

				json.dump({"IN" : INPUT, "OUT" : ""}, input_file)
				input_file.write('\n')

				ID += 1

	with open(all_status_file, 'w') as f:
		json.dump(all_status, f, indent=4)

	output_file.close()

input_file.close()


