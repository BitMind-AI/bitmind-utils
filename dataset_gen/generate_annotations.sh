#!/bin/bash

# Configuration
DATASET_BASE="google-images-holdout-deduped-commits"
TOTAL_CHUNKS=10
NUM_GPUS=10

# Hugging Face API Token
if [ -z "$1" ]; then
    echo "Hugging Face token required as the first argument."
    exit 1
fi
HF_TOKEN=$1

# Loop through each chunk and assign to a GPU
for (( i=0; i<=$TOTAL_CHUNKS; i++ )); do
    # Use modulo to distribute chunks across GPUs
    gpu_id=$(( i % NUM_GPUS ))
    
    # Run the command for this chunk - GENERATE annotations but DON'T generate images
    # Note: We don't need to specify end_index anymore, it will be determined automatically
    pm2 start generate_synthetic_dataset.py --name "annotate_${DATASET_BASE}_${i}" --no-autorestart -- \
        --hf_org 'bitmind' \
        --real_image_dataset_name "${DATASET_BASE}_${i}" \
        --diffusion_model "stabilityai/stable-diffusion-xl-base-1.0" \
        --upload_annotations \
        --hf_token "$HF_TOKEN" \
        --start_index 0 \
        --gpu_id $gpu_id
done