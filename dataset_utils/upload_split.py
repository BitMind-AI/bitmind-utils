import os
import glob
from datasets import load_dataset

HF_TOKEN = ""  # <-- Set your token here

DATASETS = [
    "bm-real",
    "celeb-a-hq",
    "ffhq-256",
    "MS-COCO-unique-256",
    "AFHQ",
    "lfw",
    "caltech-256",
    "caltech-101",
    "dtd",
    "idoc-mugshots-images"
]

for dataset_name in DATASETS:
    print(f"\n=== Processing {dataset_name} ===")
    repo_id = f"sn34-test/{dataset_name}___annotations"
    local_t2v_glob = f"test_data/annotations/{dataset_name}/*/*.json"  # Adjust as needed

    # Download existing splits
    print(f"Downloading existing dataset {repo_id} from the Hub...")
    try:
        existing = load_dataset(repo_id, token=HF_TOKEN)
    except Exception as e:
        print(f"  Could not load dataset {repo_id}: {e}")
        continue
    
    # Upload new local t2v split if available
    t2v_json_files = glob.glob(local_t2v_glob)
    if t2v_json_files:
        print(f"Uploading new t2v split from {len(t2v_json_files)} local files...")
        try:
            local_t2v = load_dataset("json", data_files=t2v_json_files, split="train")
            local_t2v.push_to_hub(repo_id, split="t2v", token=HF_TOKEN, private=True)
            print("  Uploaded new t2v split.")
        except Exception as e:
            print(f"  Could not upload new t2v split: {e}")
    else:
        print(f"  No local t2v JSON files found for {dataset_name} (looked in {local_t2v_glob}).")

print("\nAll datasets processed.")