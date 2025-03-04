
# Stage 1 - SFT微调
CUDA_VISIBLE_DEVICES=0 python -m llama_factory.train \
  --config configs/sft.yaml \
  --output_dir models/qwen2-7b-sft

# Stage 2 - DPO对齐
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  -m llama_factory.train \
  --config configs/dpo.yaml \
  --output_dir models/qwen2-7b-medical

