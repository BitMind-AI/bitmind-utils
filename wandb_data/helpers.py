from datetime import datetime
import wandb


def formatted_ts_from_epoch(ts):
    if ts is None:
        return ts
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')


def epoch_from_formatted_ts(formatted_ts):
    if formatted_ts is None:
        return formatted_ts
    return int(datetime.strptime(formatted_ts, '%Y-%m-%dT%H:%M:%S').timestamp())
