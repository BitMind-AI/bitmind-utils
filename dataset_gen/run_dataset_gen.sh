#!/bin/bash

# Example usage: ./run_dataset_generation.sh key_here

# Dataset and model configuration
REAL_DATASET='google-image-scraper'
DIFFUSION_MODEL='black-forest-labs/FLUX.1-dev'
LORA_WEIGHTS='Jovie/Midjourney'  # Optional: comment out if not using specific LoRA weights

# Total indices range
START_INDEX=0
END_INDEX=1000

# Number of GPUs
NUM_GPUS=1

# Hugging Face API Token
if [ -z "$1" ]; then
    echo "Hugging Face token required as the first argument."
    exit 1
fi
HF_TOKEN=$1

# Calculate the number of indices per GPU (add 1 for inclusive range)
RANGE_PER_GPU=$(( ($END_INDEX - $START_INDEX + 1) / $NUM_GPUS ))

# Loop to create tasks for each GPU
for (( i=0; i<$NUM_GPUS; i++ )); do
    gpu_start_index=$(( START_INDEX + i * RANGE_PER_GPU ))
    if [[ $i -eq $(( NUM_GPUS - 1 )) ]]; then
        # Last GPU takes any remainder
        gpu_end_index=$END_INDEX
    else
        gpu_end_index=$(( gpu_start_index + RANGE_PER_GPU - 1 ))
    fi

    # Build command with optional LoRA weights
    CMD="pm2 start generate_synthetic_dataset.py --name \"$REAL_DATASET $DIFFUSION_MODEL $i\" --no-autorestart -- \
        --hf_org 'bitmind' \
        --real_image_dataset_name \"$REAL_DATASET\" \
        --diffusion_model \"$DIFFUSION_MODEL\" \
        --download_annotations \
        --generate_synthetic_images \
        --upload_synthetic_images \
        --hf_token \"$HF_TOKEN\" \
        --start_index $gpu_start_index \
        --end_index $gpu_end_index \
        --gpu_id $i"

    # Add LoRA weights if specified
    if [ ! -z "$LORA_WEIGHTS" ]; then
        CMD="$CMD --lora_weights '$LORA_WEIGHTS'"
    fi

    # Execute the command
    eval $CMD
done