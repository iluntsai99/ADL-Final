evaluate_model_output_dir="model_0628_test_unseen"

for i in {0..25}
do
    python process.py $i test_unseen $evaluate_model_output_dir
    python run_summarization.py \
    --model_name_or_path model_0628 \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --train_file ./data/train_processed.jsonl \
    --test_file test_turns_unseen.jsonl \
    --max_source_length=384 \
    --max_target_length=256 \
    --val_max_target_length=256 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=1 \
    --fp16=False \
    --adafactor=True \
    --output_dir $evaluate_model_output_dir \
    --overwrite_output_dir \
    --predict_with_generate \
    --text_column IN \
    --summary_column OUT 
done
rm test_turns_unseen.jsonl
rm test_turns_unseen.jsonl.lock