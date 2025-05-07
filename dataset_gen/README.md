# BitMind Dataset Generation Tools

## Setup Instructions

1. **Clone and Setup BitMind Subnet**
```bash
git clone https://github.com/BitMind-AI/bitmind-subnet.git
cd bitmind-subnet
# Follow BitMind subnet installation instructions
# https://medium.com/bitmindlabs/start-mining-on-the-bitmind-subnet-step-by-step-runpod-tutorial-848bfa0517df
```

2. **File Integration**
```bash
# Copy files from dataset_gen to synthetic_data_generation
cp -rn /path/to/bitmind-utils/dataset_gen/* /path/to/bitmind-subnet/bitmind/synthetic_data_generation/

# Copy utils directory to bitmind/
cp -rn /path/to/bitmind-utils/utils /path/to/bitmind-subnet/bitmind/
```

3. **Update Synthetic Data Generator**
- Add the `generate_from_prompt` function from `generate_from_prompt.py` to BitMind subnet's synthetic data generator.

4. **Reinstall Requirements**
```bash
pip install -e .
```

## Tool Overview

### Dataset Generation
```bash
# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run dataset generation
./run_dataset_gen.sh YOUR_HF_TOKEN [DIFFUSION_MODEL]
```

Configure in `run_dataset_gen.sh`:
- `NUM_GPUS`: Number of available GPUs
- `DATASETS`: Array of datasets to process
- `DIFFUSION_MODEL`: Default or specified model to use

### Dataset Management Tools

1. **Generate Annotations**
```bash
./generate_annotations.sh HF_TOKEN
```

2. **Download HF Dataset**
```bash
python download_hf_dataset.py DATASET_NAME --output-dir OUTPUT_DIR --limit LIMIT --hf-token HF_TOKEN
```

3. **Upload Images to HF**
```bash
python upload_imgs_to_hf.py INPUT_DIR DATASET_NAME --hf-token HF_TOKEN [--private] [--batch-size BATCH_SIZE]
```

4. **Extract JourneyDB**
```bash
python extract_journeydb.py
```

5. **Process Split Zips**
```bash
python combine_split_zips.py SOURCE_REPO TARGET_ORG --hf-token HF_TOKEN
```

### Combining Datasets

The combine_datasets.py script allows you to merge multiple dataset chunks into a single dataset:

```bash
python combine_datasets.py ORG_NAME DATASET_NAME MODEL_NAME HF_TOKEN

# Example:
python combine_datasets.py bitmind google-images-holdout-deduped-commits_3 FLUX.1-dev YOUR_HF_TOKEN
```

This will:
1. Find all dataset chunks matching the pattern
2. Load and combine them in the correct order
3. Upload the combined dataset to Hugging Face

## Authentication

```bash
# Authenticate with Hugging Face
huggingface-cli login
```
Do not store tokens in Git.

## Best Practices

1. **Generate Annotations First**
   - Generate and upload annotations before running dataset generation
   - Reduces overall processing time
   - Use an instance with multiple GPUs to allow for parallel processing across devices

2. **RunPod GPU Recommendations**
   - Use A100 PCIE for video generation models
   - Use A40 for image generation models
   - Can utilize up to 10 GPUs on one host on RunPod (recommended)