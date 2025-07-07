#!/bin/bash

# Example usage: ./run_dataset_gen_host3.sh your_hf_token

# Check for Hugging Face API Token
if [ -z "$1" ]; then
    echo "Hugging Face token required as the first argument."
    exit 1
fi
HF_TOKEN=$1

# Set PyTorch memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Diffusion model to use
if [ -z "$2" ]; then
    DIFFUSION_MODEL=''
else
    DIFFUSION_MODEL=$2
fi

# Define the datasets to process
DATASETS=(
    ""
)
 
# Number of GPUs available on this host
NUM_GPUS=10
START_GPU=0

# Set the starting index for generation
GENERATION_START_INDEX=2000  # Start from index 2000
GENERATION_COUNT=2000        # Generate 2000 images

# Define dataset aliases
declare -A DATASET_ALIASES=(
    ["bm-real"]="bm"
    ["celeb-a-hq"]="celeb"
    ["ffhq-256"]="ffhq"
    ["MS-COCO-unique-256"]="coco"
    ["AFHQ"]="afhq"
    ["lfw"]="lfw"
    ["caltech-256"]="c256"
    ["caltech-101"]="c101"
    ["dtd"]="dtd"
    ["idoc-mugshots-images"]="mug"
)

# Configurable split for annotation loading
SPLIT=""  # Change this to your desired split (e.g., 'train', 't2i', 't2v', etc.)

# Process each dataset sequentially
for ((dataset_idx=0; dataset_idx<${#DATASETS[@]}; dataset_idx++)); do
    DATASET=${DATASETS[$dataset_idx]}
    echo "Processing dataset: $DATASET"
    
    # Get dataset size by querying the annotations dataset
    echo "Determining dataset size..."
    DATASET_SIZE=$(python -c "from datasets import load_dataset; print(len(load_dataset('sn34-test/${DATASET}___annotations', split='$SPLIT')))")
    echo "Dataset size: $DATASET_SIZE"
    
    # Check if we have enough data starting from GENERATION_START_INDEX
    AVAILABLE_INDICES=$(( DATASET_SIZE - GENERATION_START_INDEX ))
    if [ "$AVAILABLE_INDICES" -lt "$GENERATION_COUNT" ]; then
        echo "Warning: Only $AVAILABLE_INDICES indices available starting from $GENERATION_START_INDEX"
        echo "Adjusting generation count to $AVAILABLE_INDICES"
        ACTUAL_GENERATION_COUNT=$AVAILABLE_INDICES
    else
        ACTUAL_GENERATION_COUNT=$GENERATION_COUNT
    fi

    # Calculate indices per GPU (across all GPUs)
    TOTAL_GPUS=$NUM_GPUS
    INDICES_PER_GPU=$(( ($ACTUAL_GENERATION_COUNT + $TOTAL_GPUS - 1) / $TOTAL_GPUS ))
    
    # Array to store job names for waiting
    declare -a JOB_NAMES=()
    
    # Launch jobs for each GPU on this host
    for ((local_gpu=0; local_gpu<$NUM_GPUS; local_gpu++)); do
        global_gpu=$(( local_gpu + START_GPU ))
        START_INDEX=$(( GENERATION_START_INDEX + global_gpu * INDICES_PER_GPU ))
        END_INDEX=$(( START_INDEX + INDICES_PER_GPU - 1 ))
        
        # Ensure END_INDEX doesn't exceed the actual generation range
        MAX_END_INDEX=$(( GENERATION_START_INDEX + ACTUAL_GENERATION_COUNT - 1 ))
        if [ $END_INDEX -gt $MAX_END_INDEX ]; then
            END_INDEX=$MAX_END_INDEX
        fi
        
        # Skip if START_INDEX is beyond our generation range
        if [ $START_INDEX -gt $MAX_END_INDEX ]; then
            echo "Skipping GPU $local_gpu - START_INDEX $START_INDEX exceeds generation range"
            continue
        fi
        
        # Create a shorter custom name for the repository to avoid length issues
        MODEL_NAME=$(basename "$DIFFUSION_MODEL")
        DATASET_ALIAS=${DATASET_ALIASES[$DATASET]:-$DATASET}
        CUSTOM_REPO_NAME="${DATASET_ALIAS}_${MODEL_NAME}_${START_INDEX}to${END_INDEX}"
        JOB_NAME="${DATASET}_mirror_${global_gpu}"
        JOB_NAMES+=("$JOB_NAME")
        
        echo "Local GPU $local_gpu (Global GPU $global_gpu): Processing indices $START_INDEX to $END_INDEX"
        echo "Using custom repository name: $CUSTOM_REPO_NAME"
        
        # Launch the job with PM2
        pm2 start generate_synthetic_dataset.py \
            --name "$JOB_NAME" \
            --no-autorestart \
            -- \
            --hf_org "bitmind" \
            --target_org 'sn34-test' \
            --real_image_dataset_name "$DATASET" \
            --diffusion_model "$DIFFUSION_MODEL" \
            --download_annotations \
            --skip_generate_annotations \
            --generate_synthetic_images \
            --download_real_images \
            --annotation_split $SPLIT \
            --private \
            --hf_token "$HF_TOKEN" \
            --start_index $START_INDEX \
            --end_index $END_INDEX \
            --gpu_id $local_gpu \
            --output_repo_name "$CUSTOM_REPO_NAME" \
            --max_images $(( END_INDEX - START_INDEX + 1 ))
            
        # Add a small delay to prevent overwhelming the system
        sleep 2
    done
    
    # Wait for all jobs for this dataset to complete
    echo "Waiting for all jobs for $DATASET to complete..."
    
    # Wait for all jobs to finish
    for job in "${JOB_NAMES[@]}"; do
        echo "Waiting for job $job to complete..."
        
        while pm2 show "$job" | grep -q "online"; do
            echo "Job $job is still running... waiting"
            sleep 30  # Check every 30 seconds
        done
        echo "Job $job has completed"
    done
    
    # Stop and delete all jobs for this dataset
    for job in "${JOB_NAMES[@]}"; do
        pm2 delete "$job"
    done
    
    echo "All jobs for $DATASET completed."
    
    echo "Dataset $DATASET processing complete."
    echo "-------------------------------------"
done

echo "All datasets processed successfully."