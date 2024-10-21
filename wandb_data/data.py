
import wandb


def get_wanbd_runs(entity, project, start_dt, end_dt=None, validator_run_name=None):
    """ must run `wandb login` and provide your W&B api key when prompted """
    filters = {}
    if validator_run_name:
        filters["display_name"] = validator_run_name

    filters["created_at"] = {"$gte": start_dt}
    if end_dt:
        filters["created_at"]["$lte"] = end_dt

    print("Querying w&b with filters:", filters)
    api = wandb.Api()
    return api.runs(f"{entity}/{project}", filters=filters)