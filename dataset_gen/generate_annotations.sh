#!/bin/bash

# Configuration
REAL_DATASETS=(
    #"bitmind/bm-eidon-image"
    "bm-real"
    #"open-image-v7-256"
    "celeb-a-hq"
    "ffhq-256"
    "MS-COCO-unique-256"
    "AFHQ"
    "lfw"
    "caltech-256"
    "caltech-101"
    "dtd"
    "idoc-mugshots-images"
)
NUM_GPUS=10
START_INDEX=0
END_INDEX=9999
MAX_IMAGES=10000  # Set your desired maximum here

# Hugging Face API Token
if [ -z "$1" ]; then
    echo "Hugging Face token required as the first argument."
    exit 1
fi
HF_TOKEN=$1

for idx in "${!REAL_DATASETS[@]}"; do
    dataset_name="${REAL_DATASETS[$idx]}"
    gpu_id=$(( idx % NUM_GPUS ))

    # t2i/i2i prompts
    pm2 start generate_synthetic_dataset.py --name "t2i_$(basename $dataset_name)" --no-autorestart -- \
        --hf_org 'bitmind' \
        --target_org 'sn34-test' \
        --real_image_dataset_name "$dataset_name" \
        --private \
        --hf_token "$HF_TOKEN" \
        --start_index $START_INDEX \
        --end_index $END_INDEX \
        --gpu_id $gpu_id \
        --annotation_task t2i \
        --max_images $MAX_IMAGES

    # t2v/i2v prompts
    # pm2 start generate_synthetic_dataset.py --name "t2v_$(basename $dataset_name)" --no-autorestart -- \
    #     --hf_org 'bitmind' \
    #     --target_org 'sn34-test' \
    #     --real_image_dataset_name "$dataset_name" \
    #     --diffusion_model "stabilityai/stable-diffusion-xl-base-1.0" \
    #     --upload_annotations \
    #     --private \
    #     --hf_token "$HF_TOKEN" \
    #     --start_index 0 \
    #     --gpu_id $gpu_id \
    #     --annotation_task t2v \
    #     --max_images $MAX_IMAGES
done