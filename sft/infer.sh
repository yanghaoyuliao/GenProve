CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model .... \
    --max_batch_size 32 \
    --model_type qwen \
    --infer_backend vllm \
    --val_dataset ....... \
    --max_new_tokens 4096 \
    --result_path ....... \
    --temperature 0 \
    --torch_dtype float16


