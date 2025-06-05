# BitMind Dataset Generation Tools

## Setup Instructions

1. **Clone and Setup BitMind Subnet**
```bash
git clone https://github.com/BitMind-AI/bitmind-subnet.git
cd bitmind-subnet
# Follow BitMind subnet installation instructions
# https://medium.com/bitmindlabs/start-mining-on-the-bitmind-subnet-step-by-step-runpod-tutorial-848bfa0517df
```

## ⚡️ Custom BitMind Dataset Generation Pipeline (V3 Update)

### Important: Use the Custom Generation Pipeline

To enable the new output structure, mask naming, and efficient generation logic, you **must replace the default BitMind subnet pipeline** with the custom version:

```bash
cp ./bitmind-utils/dataset_gen/generation_pipeline.py ./bitmind-subnet/bitmind/generation/generation_pipeline.py
```

This ensures:
- Output files (images, videos, masks) are saved directly in the chunk directory (e.g., `.../synthetic_images/model/dataset/0_499/0.png`)
- Masks are named `<id>_mask.npy` and paired with their respective images/videos
- No per-sample `.json` files are written; if you need metadata, collect it into a single `.jsonl` file after generation
- The pipeline will **not load BLIP2** if you provide pre-generated prompts/annotations

### New Workflow: `generate_synthetic_dataset.py`

- This script orchestrates the full dataset generation process:
  1. Downloads or generates annotations (prompts)
  2. Prepares image samples (with or without prompts)
  3. Runs the custom `GenerationPipeline` for efficient batch generation
  4. Outputs are saved with clean, flat naming in the chunk directory

#### Using Pre-Generated Annotations/Prompts
- If your annotation JSONs already contain prompts, the pipeline will **skip BLIP2 loading** and use your prompts directly (saves GPU memory and time)
- This is ideal for large-scale or multi-GPU runs

#### Output Structure Example
```
synthetic_images/model_name/dataset_name/0_499/0.png
synthetic_images/model_name/dataset_name/0_499/0_mask.npy
synthetic_images/model_name/dataset_name/0_499/1.png
synthetic_images/model_name/dataset_name/0_499/1_mask.npy
...etc
```

#### Collecting Metadata
- No per-sample `.json` files are written by default
- If you need a `.jsonl` file, you can post-process the output directory to collect all metadata into a single file

### Troubleshooting & Best Practices
- **GPU Memory:** Only run one generation job per GPU at a time. BLIP2 is only loaded if prompts are missing.
- **Prompt Generation:** For large datasets, generate prompts/annotations first, then run generation jobs using those prompts.
- **Custom Logic:** Always ensure you are using the patched `generation_pipeline.py` in your BitMind subnet for the new logic to take effect.

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