"""Handle evaluation for ChemuRef dataset, for both gold and predicted anaphors
"""
from difflib import SequenceMatcher

import os
import sys
import json
import re

from utils import read_jsonl


def compute_true_positives_span_single(gold_antecedents, predicted_antecedents):

    # now compute the true positives
    tp = 0
    similarity_threshold = 1.0  # range is 0.0 - 1.0
    for predicted_antecedent in predicted_antecedents:
        for gold_antecedent in gold_antecedents:

            #  How to compare the strings?
            if (
                SequenceMatcher(None, predicted_antecedent, gold_antecedent).ratio()
                >= similarity_threshold
            ):
                tp += 1

    return tp


def compute_evaluation_metrics_span_single(outputs: dict, gold_data: dict) -> dict:

    gold_num_antecedents = 0
    predicted_num_antecedents = 0
    tp = 0
    num_examples = len(gold_data)
    num_correct = 0

    # This computation takes into account false positives error-propagation from
    # predicted anaphors. In other words, predicted_num_antecedents gets higher
    for example_id, example in outputs.items():

        gold_antecedents = list(set(example["gold_antecedents"]))
        if len(gold_antecedents) == 0:
            print("false_positives example=", example_id)
        predicted_antecedents = list(set(example["predicted_antecedents"]))

        cur_tp = compute_true_positives_span_single(
            gold_antecedents, predicted_antecedents
        )
        tp += cur_tp
        gold_num_antecedents += len(gold_antecedents)
        predicted_num_antecedents += len(predicted_antecedents)

        if cur_tp == len(gold_antecedents) == len(predicted_antecedents):
            num_correct += 1

    # This computation accounts for false negative from predicted anaphors
    # i.e. gold_num_antecedents gets higher
    for example_id, example in gold_data.items():
        if example_id not in outputs:
            print("false_negatives example=", example_id)
            gold_num_antecedents += len(example["gold_antecedents"])

    p = tp / predicted_num_antecedents
    r = tp / gold_num_antecedents
    return {
        "span_single_precision": p,
        "span_single_recall": r,
        "span_single_f1": 2 * p * r / (p + r) if (p + r) != 0 else 0,
        "strict_accuracy": num_correct / num_examples,
    }


def compute_true_positives_span_cluster(gold_antecedents, predicted_antecedents):

    # now compute the true positives
    tp = 0
    additional_predictions = 0
    similarity_threshold = 1.0  # range is 0.0 - 1.0
    count_this_cluster_before = set()
    for gold_antecedent in gold_antecedents:
        for predicted_antecedent, d in predicted_antecedents.items():

            #  How to compare the strings?
            if (
                SequenceMatcher(None, predicted_antecedent, gold_antecedent).ratio()
                >= similarity_threshold
            ):
                tp += 1
                count_this_cluster_before.add(predicted_antecedent)

            # we see if gold_antecedent match any in the cluster of this antecedent
            else:
                for span in d["cluster"]:
                    if (
                        SequenceMatcher(None, span, gold_antecedent).ratio()
                        >= similarity_threshold
                    ):
                        tp += 1
                        if predicted_antecedent in count_this_cluster_before:
                            additional_predictions += 1
                        else:
                            count_this_cluster_before.add(predicted_antecedent)

    return tp, additional_predictions


def compute_evaluation_metrics_span_cluster(outputs: dict, gold_data: dict) -> dict:

    gold_num_antecedents = 0
    predicted_num_antecedents = 0
    tp = 0

    # This computation takes into account false positives error-propagation from
    # predicted anaphors. In other words, predicted_num_antecedents gets higher
    for example_id, example in outputs.items():

        gold_antecedents = list(set(example["gold_antecedents"]))
        if len(gold_antecedents) == 0:
            print("false_positives example=", example_id)
        predicted_antecedents = example["predicted_antecedents_verbose"]

        cur_tp, additional_predictions = compute_true_positives_span_cluster(
            gold_antecedents, predicted_antecedents
        )
        tp += cur_tp
        gold_num_antecedents += len(gold_antecedents)
        predicted_num_antecedents += len(predicted_antecedents) + additional_predictions

    # This computation accounts for false negative from predicted anaphors
    # i.e. gold_num_antecedents gets higher
    for example_id, example in gold_data.items():
        if example_id not in outputs:
            print("false_negatives example=", example_id)
            gold_num_antecedents += len(example["gold_antecedents"])

    p = tp / predicted_num_antecedents
    r = tp / gold_num_antecedents
    return {
        "span_cluster_precision": p,
        "span_cluster_recall": r,
        "span_cluster_f1": 2 * p * r / (p + r) if (p + r) != 0 else 0,
    }


def main():

    gold_data_filepath = sys.argv[1]  # with gold anaphors, not predicted anaphors
    exp_dir = sys.argv[2]
    combine_methods = sys.argv[3]
    assert combine_methods in ["countBased", "noisyOR", "mixtureModel"]

    # read in the appropriate data
    gold_data = read_jsonl(gold_data_filepath)
    predictions_filepath = os.path.join(
        exp_dir, "%s_predictions.json" % combine_methods
    )
    with open(predictions_filepath, "r") as f:
        predictions = json.load(f)

    # compute evaluation metrics
    span_single_metrics = compute_evaluation_metrics_span_single(predictions, gold_data)
    span_cluster_metrics = compute_evaluation_metrics_span_cluster(
        predictions, gold_data
    )
    metrics = span_single_metrics | span_cluster_metrics

    # output metrics
    metrics_filepath = os.path.join(exp_dir, "%s_metrics.json" % combine_methods)
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()