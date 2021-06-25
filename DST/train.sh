python run_summarization.py \
    --model_name_or_path t5-base \
    --do_train=True \
    --do_eval=False \
    --do_predict=False \
    --train_file ./data/train_processed/train.jsonl \
    --max_source_length=384 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=2 \
    --fp16=False \
    --adafactor=True \
    --output_dir ./model_new1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --text_column IN \
    --summary_column OUT \
    --save_steps=3000

    #--source_prefix "summarize: " \
    #     --max_target_length=64 \