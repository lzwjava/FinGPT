#!/bin/bash
# Single GPU training script for FinGPT Benchmark with QLoRA on RTX 4070
# Usage: bash train_single_gpu.sh

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0

# Sentiment Analysis on Llama-2-7B with 4-bit QLoRA
python train_lora.py \
  --run_name sentiment-llama2-4bit \
  --base_model llama2 \
  --dataset sentiment-train \
  --test_dataset sentiment-test \
  --max_length 512 \
  --batch_size 4 \
  --gradient_steps 8 \
  --learning_rate 2e-4 \
  --num_epochs 3 \
  --log_interval 10 \
  --warmup_ratio 0.03 \
  --scheduler linear \
  --evaluation_strategy steps \
  --eval_steps 0.2 \
  --from_remote \
  --no_wandb
