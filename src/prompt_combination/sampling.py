"""The differences between sampling.py vs. count_based.py:
(1) sampling.py uses the P(z|x) computed via similarity whereas count_based.py use uniform P(z|x)
(2) for the sampling_raw_mention_probs_verbose.json, instead of storing the count 
or the probability, we store the prompt_ids that the span appears in
"""
from collections import defaultdict
import math

import os
import sys
import json
import re

import torch
import numpy as np

from prompt_combination.chemuref_verbalizer import ChemuRefVerbalizer
from utils import read_jsonl, read_json, write_json


def produce_mention_probs(predictions: dict) -> dict:
    # first produce all indicators
    all_counts = defaultdict(list)
    total_count = 0
    gold_antecedents = None
    for prompt_id, v in predictions.items():
        gold_antecedents = v["gold_antecedents"]
        for pred in v["predictions"]:
            all_counts[pred].append(prompt_id)

    # convert to list and sorted
    all_counts_lst = [(k, v) for k, v in all_counts.items()]
    all_counts_lst = sorted(all_counts_lst, key=lambda x: len(x[1]), reverse=True)

    # convert to dict
    all_counts_dict = dict()
    new_all_counts_lst = []
    for (k, v) in all_counts_lst:
        new_all_counts_lst.append(
            {
                "span": k,
                "prompts": v,
                "num_prompts": len(v),
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
            c = []
            rank = "Not predicted"
        ga_counts.append(
            {
                "span": a,
                "prompts": c,
                "num_prompts": len(c),
                "rank": rank,
            }
        )

    return {"all_counts": new_all_counts_lst, "gold_antecedents_counts": ga_counts}


def compute_prompt_probs_similar(
    predictions: dict, prompt_map: list, similarity_lst: list
) -> dict:

    # first convert similarity_lst to dict mapping id to score
    similarity_dict = dict(similarity_lst)

    # first compute \sum_i s_i. Normalized by length of prompt_id
    all_prompt_scores = dict()
    for prompt_id in prompt_map:
        # TODO: this
        prompt_str = (
            str(prompt_id) if str(prompt_id) in predictions else tuple(prompt_id)
        )
        if prompt_str in predictions:
            # assert prompt_str in predictions
            # if len(prompt_id) == 2:
            #     prompt_id = tuple(prompt_id)
            all_prompt_scores[prompt_str] = sum(
                [similarity_dict[train_id] for train_id in prompt_id]
            ) / len(prompt_id)

    # then get the list of prompt scores. Technically this step is unnecessary,
    # but put here for sanity check
    prompt_ids = list(predictions.keys())
    prompt_scores = [all_prompt_scores[prompt_id] for prompt_id in prompt_ids]

    # softmax to get the probabilities
    # prompt_probs = torch.tensor(prompt_scores).cuda().softmax(dim=-1)
    prompt_probs = torch.tensor(prompt_scores).softmax(dim=-1)

    # map it back to a dictionary
    prompt_probs = {
        prompt_id: prompt_probs[i].item() for i, prompt_id in enumerate(prompt_ids)
    }

    return prompt_probs


def compute_and_save_priors(
    example_dir, example_predictions, prompt_map, similarity_lst
):
    # check if file already exists
    priors_filepath = os.path.join(example_dir, "similar_priors.json")
    if os.path.exists(priors_filepath):
        priors = read_json(priors_filepath)
    else:

        priors = compute_prompt_probs_similar(
            example_predictions, prompt_map, similarity_lst
        )
        write_json(priors, priors_filepath)

    return priors


def sampling(mention_probs_verbose, prompt_probs, predictions):
    # 0th-step get gold_antecedents
    gold_antecedents = None
    for _, v in predictions.items():
        gold_antecedents = v["gold_antecedents"]
        break

    # Compute P(y|Z) from all the Indicator(m \in Z_i) and P(Z_i), for all i
    # NOTE: This is the crux computation-- all others are flowery
    all_probs = defaultdict(float)
    for mention_d in mention_probs_verbose["all_counts"]:
        all_probs[mention_d["span"]] = sum(
            [prompt_probs[prompt] for prompt in mention_d["prompts"]]
        )

    # convert to list and sorted
    all_probs_lst = [(k, v) for k, v in all_probs.items()]
    all_probs_lst = sorted(all_probs_lst, key=lambda x: x[1], reverse=True)

    # convert to dict
    new_all_probs_lst = []
    for (k, v) in all_probs_lst:
        new_all_probs_lst.append(
            {
                "span": k,
                "prob": v,
                "gold_antecedent": k in gold_antecedents,
            }
        )

    ga_probs = []
    for a in gold_antecedents:
        if a in all_probs:
            p = all_probs[a]
            rank = all_probs_lst.index((a, p))
        else:
            p = 0
            rank = "Not predicted"
        ga_probs.append(
            {
                "span": a,
                "prob": p,
                "rank": rank,
            }
        )

    return {"all_probs": new_all_probs_lst, "gold_antecedents_probs": ga_probs}


def combine(
    mention_probs,
    example,
    verbalizer,
    p_m_pre,
    p_m_post,
):
    context = example["context"].lower()
    anaphor = example["anaphor"].lower()
    raw_predicted_mentions = [e["span"] for e in mention_probs["all_probs"]]

    # convert mention counts to mention dict
    mentions_dict = {
        m["span"]: {
            "prob": m["prob"],
            "gold_antecedent": m["gold_antecedent"],
        }
        for m in mention_probs["all_probs"]
    }

    # filter out mentions not in context (or is context itself)
    predicted_mentions = list(
        verbalizer.filter_span(raw_predicted_mentions, context, anaphor)
    )

    # filter out mentions with prob < p_m_pre
    predicted_mentions = [
        m for m in predicted_mentions if mentions_dict[m]["prob"] >= p_m_pre
    ]

    # get clusters of spans and collapse the counts
    span_clusters = verbalizer.create_span_clusters(set(predicted_mentions), context)

    # collapsing prob
    new_prob = dict()
    for cluster_name, cluster in span_clusters.items():
        new_prob[cluster_name] = max(
            [mentions_dict[m]["prob"] for m in cluster if m in mentions_dict]
        )

    # convert to list and sorted
    new_prob_lst = [(k, v) for k, v in new_prob.items()]
    new_prob_lst = sorted(new_prob_lst, key=lambda x: x[1], reverse=True)

    # convert to dict
    final_probs_lst = []
    gold_probs_lst = []
    predictions = dict()
    for i, (k, v) in enumerate(new_prob_lst):
        isGold = bool(
            sum(
                int(mentions_dict[c]["gold_antecedent"])
                for c in span_clusters[k]
                if c in mentions_dict
            )
        )
        final_probs_lst.append(
            {
                "canonical_span": k,
                "cluster": list(span_clusters[k]),
                "prob": v,
                "contain_gold_antecedent": isGold,
            }
        )
        if isGold:
            gold_probs_lst.append(
                {
                    "canonical_span": k,
                    "cluster": span_clusters[k],
                    "prob": v,
                    "rank": i,
                }
            )
        if v >= p_m_post:
            predictions[k] = {"cluster": list(span_clusters[k]), "prob": v}

    # output the new counts
    final_golds = []
    for ga in mention_probs["gold_antecedents_probs"]:
        isNotPredicted = True
        for cur_ga in gold_probs_lst:
            if ga["span"] in cur_ga["cluster"]:
                isNotPredicted = False
                break
        if isNotPredicted:
            final_golds.append({"span": ga["span"]})

    return (
        predictions,
        {
            "all_probs": final_probs_lst,
            "gold_antecedents_probs": final_golds,
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
    similarity_map = read_json(os.path.join(exp_dir, "similarity_map.json"))
    prompt_map = read_json(os.path.join(exp_dir, "prompt_map.json"))

    # generate mention_counts
    predictions = dict()
    for example_id, example in test_data.items():

        example_dir = os.path.join(examples_dir, example_id)
        predictions_filepath = os.path.join(example_dir, "predictions.json")

        # read in predictions
        if not os.path.exists(predictions_filepath):
            print("example_id=%s does not have predictions" % example_id)
            continue
        example_predictions = read_json(predictions_filepath)

        # produce raw mention probs (\mathbb{1}(m \in Y_{z,x}))
        raw_mention_probs_verbose = produce_mention_probs(example_predictions)

        # compute P(Z_i|X) and save it
        prompt_probs = compute_and_save_priors(
            example_dir,
            example_predictions,
            prompt_map[example_id],
            similarity_map[example_id],
        )

        # print(example_id, prompt_probs)

        # produce raw mention info
        raw_mention_probs = sampling(
            raw_mention_probs_verbose, prompt_probs, example_predictions
        )

        # post-process/aggregate/ensemble the mentions
        predicted_antecedents, weighted_mention_probs = combine(
            raw_mention_probs,
            example,
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
        write_json(  # \mathbb{1}(m \ in Y_{x,z})
            raw_mention_probs_verbose,
            os.path.join(example_dir, "sampling_raw_mention_probs_verbose.json"),
        )
        write_json(  # P(y_i|z,x)
            raw_mention_probs,
            os.path.join(example_dir, "sampling_raw_mention_probs.json"),
        )
        write_json(  # filtered P(y_i|z,x)
            weighted_mention_probs,
            os.path.join(example_dir, "sampling_weighted_mention_probs.json"),
        )

    # output the predictions
    predictions_filepath = os.path.join(exp_dir, "sampling_predictions.json")
    write_json(predictions, predictions_filepath)


if __name__ == "__main__":
    main()
