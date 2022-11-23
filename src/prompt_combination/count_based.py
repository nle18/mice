from collections import defaultdict

import os
import sys
import json
import re

from prompt_combination.chemuref_verbalizer import ChemuRefVerbalizer
from utils import read_jsonl


def produce_mention_counts(predictions: dict) -> dict:

    # first produce all counts
    all_counts = defaultdict(int)
    total_count = 0
    gold_antecedents = None
    for _, v in predictions.items():
        total_count += 1
        gold_antecedents = v["gold_antecedents"]
        for pred in v["predictions"]:
            all_counts[pred] += 1

    # convert to list and sorted
    all_counts_lst = [(k, v) for k, v in all_counts.items()]
    all_counts_lst = sorted(all_counts_lst, key=lambda x: x[1], reverse=True)

    # convert to dict
    all_counts_dict = dict()
    new_all_counts_lst = []
    for (k, v) in all_counts_lst:
        new_all_counts_lst.append(
            {
                "span": k,
                "count": v,
                "prob": v / total_count,
                "gold_antecedent": k in gold_antecedents,
            }
        )

    # also get the gold antecedent counts
    ga_counts = []
    for a in gold_antecedents:
        if a in all_counts:
            c = all_counts[a]
            rank = all_counts_lst.index((a, c))
        else:
            c = 0
            rank = "Not predicted"
        ga_counts.append(
            {
                "span": a,
                "count": c,
                "prob": c / total_count,
                "rank": rank,
            }
        )

    return {"all_counts": new_all_counts_lst, "gold_antecedents_counts": ga_counts}


def countBased_combine(
    mention_counts, example, num_prompts, verbalizer, p_m_pre=0.01, p_m_post=0.1
) -> list:

    context = example["context"].lower()
    anaphor = example["anaphor"].lower()
    raw_predicted_mentions = [e["span"] for e in mention_counts["all_counts"]]

    # convert mention counts to mention dict
    mentions_dict = {
        m["span"]: {
            "count": m["count"],
            "prob": m["prob"],
            "gold_antecedent": m["gold_antecedent"],
        }
        for m in mention_counts["all_counts"]
    }

    # filter out mentions not in context (or is context itself)
    predicted_mentions = verbalizer.filter_span(
        raw_predicted_mentions, context, anaphor
    )

    # filter out mentions with prob < p_m_pre
    predicted_mentions = [
        m for m in predicted_mentions if mentions_dict[m]["prob"] > p_m_pre
    ]

    # get clusters of spans and collapse the counts
    span_clusters = verbalizer.create_span_clusters(set(predicted_mentions), context)
    # print(
    #     "example_id=%s, predicted_mentions=%s, span_clusters=%s"
    #     % (example["example_id"], predicted_mentions, span_clusters)
    # )

    # convert span_clusters to dict, with associating raw probabilities
    for cluster_name, cluster in span_clusters.items():
        span_clusters[cluster_name] = {
            m: mentions_dict[m]["prob"] for m in cluster if m in mentions_dict
        }

    # collapsing counts
    new_counts = dict()
    # total_count = 0
    for cluster_name, cluster in span_clusters.items():
        count = sum([mentions_dict[m]["count"] for m in cluster if m in mentions_dict])
        new_counts[cluster_name] = count
        # total_count += count

    # convert to list and sorted
    new_counts_lst = [(k, v) for k, v in new_counts.items()]
    new_counts_lst = sorted(new_counts_lst, key=lambda x: x[1], reverse=True)

    # convert to dict
    final_counts_lst = []
    gold_counts_lst = []
    predictions = dict()
    total_count = num_prompts
    for i, (k, v) in enumerate(new_counts_lst):
        isGold = bool(
            sum(
                int(mentions_dict[c]["gold_antecedent"])
                for c in span_clusters[k]
                if c in mentions_dict
            )
        )
        p = v / total_count
        final_counts_lst.append(
            {
                "canonical_span": k,
                "cluster": span_clusters[k],
                "count": v,
                "prob": p,
                "contain_gold_antecedent": isGold,
            }
        )
        if isGold:
            gold_counts_lst.append(
                {
                    "canonical_span": k,
                    "cluster": span_clusters[k],
                    "count": v,
                    "prob": p,
                    "rank": i,
                }
            )
        if p >= p_m_post:
            predictions[k] = {"cluster": span_clusters[k], "prob": p}

    # output the new counts
    final_golds = []
    for ga in mention_counts["gold_antecedents_counts"]:
        isNotPredicted = True
        for cur_ga in gold_counts_lst:
            if ga["span"] in cur_ga["cluster"]:
                isNotPredicted = False
                break
        if isNotPredicted:
            final_golds.append({"span": ga["span"]})

    return (
        predictions,
        {
            "all_counts": final_counts_lst,
            "gold_antecedents_counts": final_golds,
        },
    )


def main():
    exp_dir = sys.argv[1]
    test_filepath = sys.argv[2]
    p_m_pre = float(sys.argv[3])
    p_m_post = float(sys.argv[4])
    examples_dir = os.path.join(exp_dir, "examples")
    verbalizer = ChemuRefVerbalizer()
    test_data = read_jsonl(test_filepath)

    # generate mention_counts
    predictions = dict()
    for example_id, example in test_data.items():

        example_dir = os.path.join(examples_dir, example_id)
        examaple_pred_filepath = os.path.join(example_dir, "predictions.json")

        # read in predictions
        if not os.path.exists(examaple_pred_filepath):
            continue
        with open(examaple_pred_filepath, "r") as f:
            example_predictions = json.load(f)

        # produce raw mention counts
        raw_mention_counts = produce_mention_counts(example_predictions)

        # post-process/aggregate/ensemble the mentions
        num_prompts = len(example_predictions)
        predicted_antecedents, weighted_mention_counts = countBased_combine(
            raw_mention_counts,
            example,
            num_prompts,
            verbalizer,
            p_m_pre,
            p_m_post,
        )

        predictions[example_id] = {
            "context": example["context"],
            "predicted_antecedents": [k for k in predicted_antecedents],
            "gold_antecedents": [t.lower() for t in example["gold_antecedents"]],
            "predicted_antecedents_verbose": predicted_antecedents,
        }

        # outputs the counts and predictions
        mention_counts_filepath = os.path.join(
            example_dir, "countBased_raw_mention_counts.json"
        )
        with open(mention_counts_filepath, "w") as f:
            json.dump(raw_mention_counts, f, indent=4)
        weighted_mention_counts_filepath = os.path.join(
            example_dir, "countBased_weighted_mention_counts.json"
        )
        with open(weighted_mention_counts_filepath, "w") as f:
            json.dump(weighted_mention_counts, f, indent=4)

    # output the predictions
    predictions_filepath = os.path.join(exp_dir, "countBased_predictions.json")
    with open(predictions_filepath, "w") as f:
        json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    main()
