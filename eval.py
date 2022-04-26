# Provides an evaluation method for training
# Determine the accuracy of the MLM predictions
import numpy as np

from util import find_subarray

# Token sequence for `role="false"`
role_false_ids = np.array([774, 5457, 22, 3950, 22])
# Token sequence for `role="True"`
role_true_ids = np.array([774, 5457, 22, 1528, 22])

def compute_classification_metrics(labels, predicitons):
    # Find every occurance of role="true"
    true_occurances = [find_subarray(labels[i], role_true_ids) for i in range(len(labels))]
    # Find every occurance of role="false"
    false_occurances = [find_subarray(labels[i], role_false_ids) for i in range(len(labels))]

    # Offset the true and false occurances to reach the "true" or "false" token
    for i in range(len(labels)):
        true_occurances[i] += 3 # ["role", "=", '"', "true"]
        false_occurances[i] += 3 # ["role", "=", '"', "true"]

        # Assert that the alignment has been performed properly
        assert np.all(np.array(labels[i])[true_occurances[i]] == 1528)
        assert np.all(np.array(labels[i])[false_occurances[i]] == 3950)

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for sample, true_occur, false_occur in zip(predicitons, true_occurances, false_occurances):
        true_preds = sample[true_occur] == 1528
        false_preds = sample[false_occur] == 3950

        sample_true_pos = int(np.sum(true_preds))
        sample_false_pos = int(np.sum(false_preds))

        true_pos += sample_true_pos
        false_pos += sample_false_pos
        true_neg += len(true_preds) - sample_true_pos
        false_neg += len(false_preds) - sample_false_pos
    
    # Compute the accuracy
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    # Compute the precision
    precision = true_pos / (true_pos + false_pos)
    # Compute the recall
    recall = true_pos / (true_pos + false_neg)
    # Compute the F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }