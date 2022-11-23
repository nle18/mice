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


def compute_P_m_give_Zi(logits, logits_info):
    # first we need to use logits_info to map each generated span to the appropriate token_ids
    mentions = defaultdict(dict)
    for prompt_id, d in logits_info.items():
        raw_mentions = d["output_text"].split(" |")
        for i, (id, token) in enumerate(d["first_tokens"]):
            mention_name = raw_mentions[i].strip().lower()
            if mention_name not in mentions:
                mentions[mention_name] = {
                    "meta": {
                        "raw_form": raw_mentions[i],
                        "first_token": token,
                        "first_token_id": id,
                    }
                }

    # second we softmax the logits
    probs = dict()
    for prompt in logits:
        probs[prompt] = (
            torch.tensor(logits[prompt]).cuda().softmax(dim=-1)
        )  # dim (*, vocab_size)

        # probs[prompt] = torch.tensor(logits[prompt]).softmax(
        #     dim=-1
        # )  # dim (*, vocab_size)

    # third we find the appropriate prob for each prompt Z_i, for each mention m
    # eg P(m | Z_i). This is done by maxing over the first token of m over each
    # first position in the prediction
    for mention in mentions:
        token_id = mentions[mention]["meta"]["first_token_id"]
        mentions[mention]["probs"] = dict()
        for prompt, prob in probs.items():
            all_pos_probs = [prob[i, token_id].item() for i in range(prob.shape[0])]
            mentions[mention]["probs"][prompt] = max(all_pos_probs)

    return mentions


def directModel(mention_probs_verbose, predictions):

    # 0th-step get gold_antecedents
    gold_antecedents = None
    for _, v in predictions.items():
        gold_antecedents = v["gold_antecedents"]
        break

    # first from all the P(m|Z_i), compute a single P(m|Z_1,...,Z_i) for each m
    # P(m|Z_1,...,Z_i) = \product_i P(m|Z_i)
    # perform in log space -> log P(m|Z_1,...,Z_i) = \sum_i log P(m|Z_i)
    # NOTE: This is the crux computation-- all others are flowery
    all_probs = defaultdict(float)
    for mention, mention_d in mention_probs_verbose.items():
        all_probs[mention] = sum(
            [np.log(prob) for _, prob in mention_d["probs"].items()]
        )

    # convert to list and sorted
    all_probs_lst = [(k, v) for k, v in all_probs.items()]
    all_probs_lst = sorted(all_probs_lst, key=lambda x: x[1])

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
    new_logprob = dict()
    for cluster_name, cluster in span_clusters.items():
        new_logprob[cluster_name] = max(
            [mentions_dict[m]["prob"] for m in cluster if m in mentions_dict]
        )

    # convert to list and sorted
    new_logprob_lst = [(k, v) for k, v in new_logprob.items()]
    new_logprob_lst = sorted(new_logprob_lst, key=lambda x: x[1], reverse=True)

    # convert to dict
    final_probs_lst = []
    gold_probs_lst = []
    predictions = dict()
    for i, (k, v) in enumerate(new_logprob_lst):
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
            "all_logprobs": final_probs_lst,
            "gold_antecedents_logprobs": final_golds,
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
    prompt_map = read_json(os.path.join(exp_dir, "prompt_map.json"))

    # generate mention_counts
    predictions = dict()
    for example_id, example in test_data.items():

        try:

            example_dir = os.path.join(examples_dir, example_id)
            predictions_filepath = os.path.join(example_dir, "predictions.json")

            # read in predictions and logits file
            if not os.path.exists(predictions_filepath):
                print("example_id=%s does not have predictions" % example_id)
                continue
            example_predictions = read_json(predictions_filepath)

            # compute P(m|Z_i)
            mention_probs_verbose_filepath = os.path.join(
                example_dir, "direct_raw_mention_probs_verbose.json"
            )
            if os.path.exists(mention_probs_verbose_filepath):
                with open(mention_probs_verbose_filepath, "r") as f:
                    raw_mention_probs_verbose = json.load(f)

            else:
                with open(os.path.join(example_dir, "logits_info.json"), "r") as f:
                    logits_info = json.load(f)
                print("logits_filepath=", os.path.join(example_dir, "logits.npz"))
                logits = np.load(os.path.join(example_dir, "logits.npz"))
                # map each span to its LM probabilities
                raw_mention_probs_verbose = compute_P_m_give_Zi(logits, logits_info)
                write_json(raw_mention_probs_verbose, mention_probs_verbose_filepath)

            # produce raw mention info
            raw_mention_probs = directModel(
                raw_mention_probs_verbose, example_predictions
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
            mention_probs_filepath = os.path.join(
                example_dir, "direct_raw_mention_probs.json"
            )
            write_json(raw_mention_probs, mention_probs_filepath)
            weighted_mention_probs_filepath = os.path.join(
                example_dir, "direct_weighted_mention_probs.json"
            )
            write_json(weighted_mention_probs, weighted_mention_probs_filepath)

        except:

            print(
                "Something wrong with example_id=%s; most likely corrupted logits.npz files"
                % example_id
            )
            predictions[example_id] = {
                "context": example["context"],
                "predicted_antecedents": [],
                "gold_antecedents": [t.lower() for t in example["gold_antecedents"]],
                "predicted_antecedents_verbose": dict(),
            }

    # output the predictions
    predictions_filepath = os.path.join(exp_dir, "direct_predictions.json")
    write_json(predictions, predictions_filepath)


if __name__ == "__main__":
    main()
