from tqdm.auto import tqdm
import re
from scipy.spatial import distance
import sys
split = sys.argv[1]  # Choose from dev, test_seen, test_unseen

def LCSubStr(X, Y, m, n):
 
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains the
    # length of longest common suffix of
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program.
 
    # LCSuff is the table with zero
    # value initially in each cell
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
 
    # To store the length of
    # longest common substring
    result = 0
 
    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result

import json
import os

if split == "dev":
    status_dir = "dev_status.json"
    data_dir = "./data/dev"
elif split == "test_seen":
    status_dir = "test_status_seen.json"
    data_dir = "./data/test_seen"
elif split == "test_unseen":
    status_dir = "test_status_unseen.json"
    data_dir = "./data/test_unseen" 

with open(status_dir, 'r') as f:
    results = json.load(f)

services_list = {}
category_list = {}

schema_dir = "./data/schema.json"

with open(schema_dir) as f:
    schema = json.load(f)
    for service in schema:
        slots = []
        for slot in service["slots"]:
            slots.append(slot["name"])
            if slot["is_categorical"] == True:
                category_list[service["service_name"] + '-' + slot["name"]] = [value.lower() for value in slot["possible_values"]]
        services_list[service["service_name"]] = slots

final_answers = {}

glove = {}
with open('./glove.6B.50d.txt') as fp:
    row1 = fp.readline()
    # if the first row is not header
    if not re.match("^[0-9]+ [0-9]+$", row1):
        # seek to 0
        fp.seek(0)
    # otherwise ignore the header

    for i, line in tqdm(enumerate(fp)):
        cols = line.rstrip().split(" ")
        word = cols[0]
        vector = [float(v) for v in cols[1:]]

        glove[word] = vector
        glove_dim = len(vector)

filenames = sorted(os.listdir(data_dir))
for file_name in filenames:
    file_dir = os.path.join(data_dir, file_name)
    
    with open(file_dir, 'r') as f:
        data = json.load(f)
        
        for i, dialogue in enumerate(data):
            final_answer = {}

            dialogue_services = dialogue["services"]
            
            predict_answer = results[dialogue["dialogue_id"]]["status"].replace("<extra_id_3>", '').strip()
            predict_answer = predict_answer.split('<extra_id_5>')

            for answer in predict_answer:
                if not answer or '<extra_id_4>' not in answer:
                    continue

                slot_name = answer.split('<extra_id_4>')[0].strip()
                slot_value = answer.split('<extra_id_4>')[1].strip()
                
                if slot_value == '<extra_id_6>' or not slot_value:
                    continue

                slot_full_name = ''

                # Same slot_name may belong to more than 1 service (Given that service name is not put into training) 
                for service in dialogue_services:
                    if slot_name in services_list[service]:
                        slot_full_name = service + '-' + slot_name
                        if slot_full_name not in final_answer:
                            if slot_full_name in category_list and slot_value.lower() not in category_list[slot_full_name]:
                                if slot_value.lower() != 'dontcare': # Keep "dontcare" since it is also an option (although not listed)
                                    # Replace value if Maximum Overlap Substring has length >= 2
                                    maxi = 2
                                    new_slot_value = ''
                                    for value in category_list[slot_full_name]:
                                        overlap = LCSubStr(slot_value.lower(), value, len(slot_value), len(value))
                                        if overlap >= maxi:
                                            maxi = overlap
                                            new_slot_value = value
                                    # If no Overlap Substring, just choose the first answer
                                    # (Use glove to compare similarity can improve a little bit e.g. replace five with 5, but I am lazy)
                                    if not new_slot_value:
                                        # new_slot_value = category_list[slot_full_name][0]
                                        try:
                                            min_distance = 1
                                            for value in category_list[slot_full_name]:
                                                cosine_distance = distance.cosine(glove[slot_value.lower()],glove[value.lower()])
                                                if cosine_distance < min_distance:
                                                    min_distance = cosine_distance
                                                    new_slot_value = value
                                        except:
                                            new_slot_value = category_list[slot_full_name][0]
                                        
                                    slot_value = new_slot_value
                            final_answer[slot_full_name] = slot_value.lower()
                        else:
                            # Repeated slot_name : slot_value pair
                            continue
                if slot_full_name == '':
                    continue

            final_answers[dialogue["dialogue_id"]] = final_answer

def write_csv(ans, output_path):
    ans = sorted(ans.items(), key=lambda x: x[0])
    with open(output_path, 'w') as f:
        f.write('id,state\n')
        for dialogue_id, states in ans:
            if len(states) == 0:  # no state ?
                str_state = 'None'
            else:
                states = sorted(states.items(), key=lambda x: x[0])
                str_state = ''
                for slot, value in states:
                    # NOTE: slot = "{}-{}".format(service_name, slot_name)
                    str_state += "{}={}|".format(
                            slot.lower(), value.replace(',', '_').lower())
                str_state = str_state[:-1]
            f.write('{},{}\n'.format(dialogue_id, str_state))

if split == "dev":
    with open("./data/dev_answer.json", 'r') as f:
        answers = json.load(f)

    correct = total = 0

    for dialogue_id in answers:
        if answers[dialogue_id] == final_answers[dialogue_id]:
            correct += 1
        else:
            print(dialogue_id)
            print(answers[dialogue_id])
            print(final_answers[dialogue_id])
            print()
            pass
        total += 1

    print("correct:", correct, "total:", total, "=", correct / total, "%")

elif split == "test_seen":
    write_csv(final_answers, 'seen_submission.csv')
elif split == "test_unseen":
    write_csv(final_answers, 'unseen_submission.csv')