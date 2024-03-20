DATA_PATH=""
OUTPUT_PATH=""
MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
MODEL_PATH="../deepseek-ai/deepseek-coder-6.7b-instruct"

cd finetune && nohup deepspeed --include localhost:1 finetune_lora.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1562 \
    --save_total_limit 100 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 False