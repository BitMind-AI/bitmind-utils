#!/bin/bash

# Example usage: ./run_dataset_gen.sh your_hf_token

# Check for Hugging Face API Token
if [ -z "$1" ]; then
    echo "Hugging Face token required as the first argument."
    exit 1
fi
HF_TOKEN=$1

# Diffusion model to use
DIFFUSION_MODEL='stabilityai/stable-diffusion-xl-base-1.0'

# Define the datasets to process
DATASETS=(
    "google-images-holdout-deduped-commits_3"
    #"google-images-holdout-deduped-commits_4"
    #"google-images-holdout-deduped-commits_5"
)

# Number of GPUs available
NUM_GPUS=10  # Adjust based on your system

# Process each dataset sequentially
for ((dataset_idx=0; dataset_idx<${#DATASETS[@]}; dataset_idx++)); do
    DATASET=${DATASETS[$dataset_idx]}
    echo "Processing dataset: $DATASET"
    
    # Get dataset size by querying the annotations dataset
    echo "Determining dataset size..."
    DATASET_SIZE=$(python -c "from datasets import load_dataset; print(len(load_dataset('bitmind/${DATASET}___annotations', split='train')))")
    echo "Dataset size: $DATASET_SIZE"
    
    # Calculate indices per GPU
    INDICES_PER_GPU=$(( ($DATASET_SIZE + $NUM_GPUS - 1) / $NUM_GPUS ))
    
    # Array to store job names for waiting
    declare -a JOB_NAMES=()
    
    # Launch jobs for each GPU
    for ((gpu_id=0; gpu_id<$NUM_GPUS; gpu_id++)); do
        START_INDEX=$(( gpu_id * INDICES_PER_GPU ))
        END_INDEX=$(( START_INDEX + INDICES_PER_GPU - 1 ))
        
        # Ensure END_INDEX doesn't exceed dataset size
        if [ $END_INDEX -ge $DATASET_SIZE ]; then
            END_INDEX=$(( DATASET_SIZE - 1 ))
        fi
        
        JOB_NAME="${DATASET}_mirror_${gpu_id}"
        JOB_NAMES+=("$JOB_NAME")
        
        echo "GPU $gpu_id: Processing indices $START_INDEX to $END_INDEX"
        
        # Launch the job with PM2
        pm2 start generate_synthetic_dataset.py \
            --name "$JOB_NAME" \
            --no-autorestart \
            -- \
            --hf_org "bitmind" \
            --real_image_dataset_name "$DATASET" \
            --diffusion_model "$DIFFUSION_MODEL" \
            --download_annotations \
            --skip_generate_annotations \
            --generate_synthetic_images \
            --upload_synthetic_images \
            --hf_token "$HF_TOKEN" \
            --start_index $START_INDEX \
            --end_index $END_INDEX \
            --gpu_id $gpu_id
            
        # Add a small delay to prevent overwhelming the system
        sleep 2
    done
    
    # Wait for all jobs for this dataset to complete
    echo "Waiting for all jobs for $DATASET to complete..."
    
    # Wait for all jobs to finish
    for job in "${JOB_NAMES[@]}"; do
        echo "Waiting for job $job to complete..."
        
        # Replace with this polling approach:
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