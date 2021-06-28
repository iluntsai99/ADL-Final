Preprocess
train_process.py:		preprocess original training dataset to "summarization" format and store in data/train_processed.jsonl
dev_to_test_format.py:	transform dev set to same format as test set and store in data/dev.json
dev_generate_answer.py:	generate correct answer for dev set and store in data/dev_answer.json 

Training
train.sh using run_summarization.py

Validation / Testing
Step 1:
dev.sh / test.sh / test_unseen.sh
using process.py and run_summarization.py to generate dev_status.json / test_status_seen.json / test_status_unseen.json
Step 2:
final_answer_process.py to generate submisison file or validation
(change to dev / test_seen / test_unseen inside the file to switch)

