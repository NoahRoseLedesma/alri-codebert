# Provides an evaluation method for training
# Determine the accuracy of the MLM predictions
import numpy as np
import torch

from util import find_subarray

# Token sequence for `role="false"`
role_false_ids = np.array([774, 5457, 22, 3950, 22])
# Token sequence for `role="True"`
role_true_ids = np.array([774, 5457, 22, 1528, 22])

total_true_pos = 0
total_false_pos = 0
total_true_neg = 0
total_false_neg = 0

# This function is very much a hack to compute metrics on-GPU without
# having to moving data from GPU to CPU back to GPU.
# Note that results are sent to main memory before
# compute_classification_metrics is called in the evaluation loop.
def preprocess_logits_for_metrics(logits, labels):
    global total_true_pos, total_false_pos, total_true_neg, total_false_neg

    predictions = torch.argmax(logits, axis=-1)

    # Flattern the predictions and labels
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Find the indidicies of all the true and false occurances in the labels
    true_occurance_indicies = (labels == 1528).nonzero()
    false_occurance_indicies = (labels == 3950).nonzero()

    true_predictions = predictions[true_occurance_indicies] == labels[true_occurance_indicies]
    false_preditions = predictions[false_occurance_indicies] == labels[false_occurance_indicies]

    # Confusion matrix
    true_pos = int(true_predictions.sum())
    false_pos = int(false_preditions.sum())
    true_neg = len(true_predictions) - true_pos
    false_neg = len(false_preditions) - false_pos

    # Contribute to the total metrics
    total_true_pos += true_pos
    total_false_pos += false_pos
    total_true_neg += true_neg
    total_false_neg += false_neg

    return logits

def compute_classification_metrics(_):
    # Compute the accuracy using total true and false positives and negatives
    try:
        accuracy = (total_true_pos + total_true_neg) / (total_true_pos + total_true_neg + total_false_pos + total_false_neg)
    except ZeroDivisionError:
        accuracy = 0

    # Compute the precision using total true positives and false positives
    try:
        precision = total_true_pos / (total_true_pos + total_false_pos)
    except ZeroDivisionError:
        precision = 0
    
    # Compute the recall using total true positives and false negatives
    try:
        recall = total_true_pos / (total_true_pos + total_false_neg)
    except ZeroDivisionError:
        recall = 0

    # Compute the F1 score using precision and recall
    try:
        f1_score = 2 * precision * recall / (precision + recall)
    except(ZeroDivisionError):
        f1_score = 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "true_pos": total_true_pos,
        "false_pos": total_false_pos,
        "true_neg": total_true_neg,
        "false_neg": total_false_neg
    }