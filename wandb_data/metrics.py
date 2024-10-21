

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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