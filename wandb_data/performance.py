from collections import defaultdict
import pandas as pd
import os

from metrics import compute_metrics


def compute_miner_performance(
        wandb_validator_runs,
        miner_uid=None, 
        start_ts=None,
        end_ts=None,
        download_images=False,
        download_videos=False,
        download_dest='',
        validator_run_name=None):

    challenge_data = defaultdict(list)
    for run in wandb_validator_runs:
            
        if validator_run_name is not None and run.name != validator_run_name: 
            continue

        history_df = run.history()
        for _, challenge_row in history_df.iterrows():
            if start_ts is not None and challenge_row['_timestamp'] < start_ts:
                continue
            if end_ts is not None and challenge_row['_timestamp'] > end_ts:
                continue

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
            
            modality = challenge_row.get('modality', 'image')
            label = challenge_row['label']

            if download_images and modality == 'image':
                try:
                    image_path = challenge_row['image']['path']
                    print(f"Downloading challenge image {image_path}")
                    if not os.path.exists(image_path):
                        run.file(image_path).download(os.path.join(download_dest, os.path.basename(image_path)))
                except Exception as e:
                    print(f'Failed to download image: {e}')

            if download_videos and modality == 'video':
                try:
                    video_path = challenge_row['video']['path']
                    print(f"Downloading challenge video {video_path}")
                    if not os.path.exists(video_path):
                        run.file(video_path).download(os.path.join(download_dest, os.path.basename(video_path)))
                except Exception as e:
                    print(f'Failed to download video: {e}')
            
            # record predictions and labels for each miner
            for pred, uid in zip(miner_preds, miner_uids):
                if miner_uid is not None and uid != miner_uid:
                    continue
                if pred == -1:
                    continue
                challenge_data['modality'].append(modality)
                challenge_data['uid'].append(uid)
                challenge_data['prediction'].append(pred)
                challenge_data['label'].append(label)

                wandb_path = challenge_row[modality]['path']
                challenge_data['wandb_filepath'].append(wandb_path)
                local_path = os.path.join(download_dest, os.path.basename(wandb_path))
                local_path = local_path if os.path.exists(local_path) else 'not downloaded'
                challenge_data['local_filepath'].append(local_path)


    all_miner_preds_df = pd.DataFrame(challenge_data)

    # Compute performance metrics for each miner
    miner_perf_data = []
    for uid, miner_preds in all_miner_preds_df.groupby('uid'):
        for modality in ['image', 'video']:
            miner_modality_preds = miner_preds[miner_preds['modality'] == modality]
            if len(miner_modality_preds) > 0:
                metrics = compute_metrics(
                    miner_modality_preds['prediction'].tolist(), 
                    miner_modality_preds['label'].tolist())
                metrics['uid'] = uid
                metrics['modality'] = modality
                miner_perf_data.append(metrics)
    
    miner_perf_df = pd.DataFrame(miner_perf_data)
    return {'predictions': all_miner_preds_df, 'performance': miner_perf_df}


