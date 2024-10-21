from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import defaultdict
import pandas as pd
import os


def compute_miner_performance(
        wandb_validator_runs,
        miner_uid=None, 
        start_ts=None,
        end_ts=None,
        download_fake_images=False,
        validator_run_name=None):

    miner_perf = defaultdict(lambda: {'predictions': [], 'labels': []})
    fake_image_preds = defaultdict(dict)
    for run in wandb_validator_runs:
            
        if validator_run_name is not None and run.name != validator_run_name: 
            continue

        history_df = run.history()
        image_files = [f for f in run.files() if f.name.endswith(".png")]
        for i, challenge_row in history_df.iterrows():
            if start_ts is not None and challenge_row['_timestamp'] < start_ts:
                continue
            if end_ts is not None and challenge_row['_timestamp'] > end_ts:
                continue

            label = challenge_row['label']

            try:
                miner_preds = challenge_row['pred']
            except KeyError as e:
                miner_preds = challenge_row['predictions']
                
            try:
                miner_uids = challenge_row['miner_uid']
            except KeyError as e:
                miner_uids = challenge_row['miner_uids']
                
            if isinstance(miner_uids, dict):  
                continue  # ignore improperly formatted instances

            # record predictions and labels for each miner
            for pred, uid in zip(miner_preds, miner_uids):
                if miner_uid is not None and uid != miner_uid:
                    continue
                         
                # record synthetic images and predictions (real images are not logged)
                if download_fake_images:
                    try:
                        image_path = challenge_row['image']['path']
                        challenge_image = [f for f in image_files if f.name == image_path][0]
                        if not os.path.exists(challenge_image.name):
                            challenge_image.download()
                        fake_image_preds[image_path][uid] = pred
                    except Exception as e:
                        pass

                if pred == -1:
                    continue
                miner_perf[uid]['predictions'].append(pred)
                miner_perf[uid]['labels'].append(label)
                            
    metrics = {uid: compute_metrics(data['predictions'], data['labels']) for uid, data in miner_perf.items()}
    flattened_metrics = []
    for uid, metric_dict in metrics.items():
        flattened_metrics.append({'uid': uid, **metric_dict})
    metrics_df = pd.DataFrame(flattened_metrics)

    return metrics_df, fake_image_preds


def compute_metrics(predictions, labels):
    """
    Compute accuracy, precision, recall, and F1 score from predictions and labels.

    Parameters:
    predictions (list): List of predicted values
    labels (list): List of true labels

    Returns:
    dict: A dictionary with accuracy, precision, recall, and F1 score
    """
    accuracy = accuracy_score(labels, [1 if p > 0.5 else 0 for p in predictions])
    precision = precision_score(labels, [1 if p > 0.5 else 0 for p in predictions], zero_division=0)
    recall = recall_score(labels, [1 if p > 0.5 else 0 for p in predictions], zero_division=0)
    f1 = f1_score(labels, [1 if p > 0.5 else 0 for p in predictions], zero_division=0)
    roc_auc = roc_auc_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc,
        "sample_size": len(predictions)
    }