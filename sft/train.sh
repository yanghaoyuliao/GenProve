CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model ./Qwen/Qwen3-4B \
    --train_type full \
    --dataset ./sft/train/data_v2.jsonl\
    --torch_dtype float16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --save_steps 50 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir ./models/sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --report_to swanlab \
    --swanlab_project swift-robot \
    --deepspeed zero3 \
    --gradient_checkpointing True\
    # --lora_rank 8 \
    # --lora_alpha 16 \