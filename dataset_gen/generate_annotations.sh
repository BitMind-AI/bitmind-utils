#!/bin/bash

# Configuration
REAL_DATASETS=(
    "internet-images"
    #"bitmind/bm-eidon-image"
    # "bm-real"
    #"open-image-v7-256"
    # "celeb-a-hq"
    # "ffhq-256"
    # "MS-COCO-unique-256"
    # "AFHQ"
    # "lfw"
    # "caltech-256"
    # "caltech-101"
    # "dtd"
    # "idoc-mugshots-images"
)
NUM_GPUS=10
GLOBAL_START_INDEX=10000  # Renamed to avoid confusion
GLOBAL_END_INDEX=19999    # Renamed to avoid confusion
MAX_IMAGES=10000  # Set your desired maximum here

# Hugging Face API Token
if [ -z "$1" ]; then
    echo "Hugging Face token required as the first argument."
    exit 1
fi
HF_TOKEN=$1

if [ "${#REAL_DATASETS[@]}" -eq 1 ]; then
    # Only one dataset: split across all GPUs with offset
    dataset_name="${REAL_DATASETS[0]}"
    IMAGES_PER_GPU=$(( (MAX_IMAGES + NUM_GPUS - 1) / NUM_GPUS ))  # Round up
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        # Calculate relative start/end within the chunk
        RELATIVE_START=$(( gpu_id * IMAGES_PER_GPU ))
        RELATIVE_END=$(( RELATIVE_START + IMAGES_PER_GPU - 1 ))
        if [ $RELATIVE_END -ge $MAX_IMAGES ]; then
            RELATIVE_END=$((MAX_IMAGES - 1))
        fi
        
        # Add the global offset
        ACTUAL_START=$(( GLOBAL_START_INDEX + RELATIVE_START ))
        ACTUAL_END=$(( GLOBAL_START_INDEX + RELATIVE_END ))
        
        echo "GPU $gpu_id: Processing images $ACTUAL_START to $ACTUAL_END"
        
        pm2 start generate_synthetic_dataset.py --name "t2i_${dataset_name}_gpu${gpu_id}" --no-autorestart -- \
            --local_image_dir "$dataset_name" \
            --hf_org 'bitmind' \
            --target_org 'sn34-test' \
            --real_image_dataset_name "$dataset_name" \
            --private \
            --hf_token "$HF_TOKEN" \
            --start_index $ACTUAL_START \
            --end_index $ACTUAL_END \
            --gpu_id $gpu_id \
            --annotation_task t2i \
            --max_images $MAX_IMAGES
    done
else
    # Multiple datasets: use global start/end for each
    for idx in "${!REAL_DATASETS[@]}"; do
        dataset_name="${REAL_DATASETS[$idx]}"
        gpu_id=$(( idx % NUM_GPUS ))
        pm2 start generate_synthetic_dataset.py --name "t2i_$(basename $dataset_name)" --no-autorestart -- \
            --local_image_dir "$dataset_name" \
            --hf_org 'bitmind' \
            --target_org 'sn34-test' \
            --real_image_dataset_name "$dataset_name" \
            --private \
            --hf_token "$HF_TOKEN" \
            --start_index $GLOBAL_START_INDEX \
            --end_index $GLOBAL_END_INDEX \
            --gpu_id $gpu_id \
            --annotation_task t2i \
            --max_images $MAX_IMAGES
    done
fi

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