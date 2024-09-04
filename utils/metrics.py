

def compute_metrics(tp=0, fp=0, tn=0, fn=0):
    """
    Compute precision, recall, and F1 score from tp, fp, fn.

    Parameters:
    tp (int): True positives
    fp (int): False positives
    fn (int): False negatives

    Returns:
    dict: A dictionary with precision, recall, and F1 score
    """
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {
        "accuracy": (tp + tn) / (tp + fp + tn + fn),
        "precision": prec,
        "recall": rec,
        "f1_score": 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0,
        "sample_size": tp + tn + fp + fn
    }
