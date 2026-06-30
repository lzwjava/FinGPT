#!/bin/bash
# Single GPU training script for FinGPT Forecaster with QLoRA on RTX 4070
# Usage: bash train_single_gpu.sh

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0

# Forecaster on Llama-2-7B with 4-bit QLoRA, long context
python train_lora.py \
  --run_name forecaster-llama2-4bit \
  --base_model llama2 \
  --dataset dow30-20230601-20230930-llama \
  --max_length 2048 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --log_interval 10 \
  --warmup_ratio 0.03 \
  --scheduler constant \
  --evaluation_strategy steps \
  --eval_steps 0.2 \
  --from_remote \
  --no_wandb
