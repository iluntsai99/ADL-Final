python run_summarization.py \
    --model_name_or_path t5-base \
    --do_train=True \
    --do_eval=False \
    --do_predict=False \
    --train_file ./data/train_processed.jsonl \
    --max_source_length=384 \
    --max_target_length=256 \
    --val_max_target_length=256 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=1 \
    --fp16=False \
    --adafactor=True \
    --output_dir ./model_0628 \
    --overwrite_output_dir \
    --predict_with_generate \
    --text_column IN \
    --summary_column OUT \
    --save_steps=3000