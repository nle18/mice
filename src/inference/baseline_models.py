"""Models to try 
- recent-2
- recent-3
- recent-4
- recent-5
- recent-all
- random (binomial)
"""
from difflib import SequenceMatcher
from pydoc import doc
from typing import List

import os
import sys
import json
import random

from evaluations.chemuref_evaluation import compute_evaluation_metrics_span_single
from utils import *

SEED = 30318
random.seed(SEED)

################################################################################
# START CODE: This code is for alignment between conll-tokenized text and original text
# Most of the code here borrowed from ChEMU-Ref/brat.py
################################################################################
TOKEN_PATH = "/srv/share5/nghia6/data/ChEMU-Ref/chemu2021.anaphora_resolution.dev.conll"
TEXT_PATH = "/srv/share5/nghia6/data/ChEMU-Ref/chemu2021.anaphora_resolution.dev/"

# TOKEN_PATH = (
#     "/srv/share5/nghia6/data/ChEMU-Ref/chemu2021.anaphora_resolution.train.conll"
# )
# TEXT_PATH = "/srv/share5/nghia6/data/ChEMU-Ref/chemu2021.anaphora_resolution.train/"


def align_span_and_token_from_tokenized_conll(text_path, token_path):
    text = None
    with open(text_path, "r") as fr:
        text = fr.read()
    span_and_token = []
    with open(token_path, "r") as fr:
        tokens = fr.readlines()
        span_start_index = 0
        token_index = 0
        for token in tokens:
            token = token.split("\n")[0]
            if len(token) == 0:
                span_start_index += 1
                span_and_token.append([])  # [] means the end of the sentence
                continue

            find_start = False
            while find_start == False:
                if text[span_start_index : span_start_index + 1] == token[0]:
                    find_start = True
                if find_start == False:
                    span_start_index += 1
                    continue
                find_token = False
                for span_end_index in range(span_start_index + 1, len(text) + 1):
                    if str(text[span_start_index:span_end_index]) == str(token):
                        span_and_token.append(
                            [span_start_index, span_end_index, token, token_index]
                        )
                        token_index += 1
                        span_start_index = span_end_index
                        find_token = True
                        break
                if find_token == False:
                    print("error. can not find the tokne.", token, span_start_index)

        #     print(span_and_token)
        assert len(span_and_token) == len(tokens)

    return span_and_token  # [span_index_start, span_index_end, txt, token_id]


def get_spanTokenAlignment_and_originalText(doc_key: str) -> str:

    original_gold_example_brat_txt_path = os.path.join(TEXT_PATH, "%s.txt" % doc_key)
    token_path_per = os.path.join(TOKEN_PATH, "%s.conll" % doc_key)

    span_and_tokens = align_span_and_token_from_tokenized_conll(
        original_gold_example_brat_txt_path, token_path_per
    )
    original_gold_example_brat_txt_path = os.path.join(TEXT_PATH, "%s.txt" % doc_key)
    with open(original_gold_example_brat_txt_path, "r") as fr:
        original_gold_example_brat_txt = fr.read()
    return (span_and_tokens, original_gold_example_brat_txt)


def get_token_to_span_index_start(span_and_tokens, token_index):
    for st in span_and_tokens:
        if len(st) == 0:
            continue
        if st[3] == token_index:
            return st[0]


def get_token_to_span_index_end(span_and_tokens, token_index):
    for st in span_and_tokens:
        if len(st) == 0:
            continue
        if st[3] == token_index:
            return st[1]


def get_original_text_from_conll_tokenized_span(
    doc_key, mention, span_and_tokens, original_text
):

    predicted_span_start = get_token_to_span_index_start(span_and_tokens, mention[0])
    predicted_span_end = get_token_to_span_index_start(span_and_tokens, mention[1] + 1)
    return original_text[predicted_span_start:predicted_span_end].strip()


################################################################################
# END CODE
################################################################################
def find_candidate_antecedents(
    doc_key,
    anaphor: List[int],
    predicted_mentions: List[List[int]],
    span_and_tokens,
    original_text,
) -> List[str]:
    candidates = []
    for mention in predicted_mentions:
        if mention[0] < anaphor[0]:
            [start, end] = mention
            candidate_text = get_original_text_from_conll_tokenized_span(
                doc_key, mention, span_and_tokens, original_text
            )
            candidates.append(candidate_text)

    return candidates


def get_anaphor_tokenIDs_from_text(
    original_text, trimmed_text, anaphor_span, span_and_tokens
):

    # first find the span start and end
    span_start = original_text.find(
        anaphor_span, len(trimmed_text) - len(anaphor_span) - 3
    )
    # print(span_start, anaphor_span)
    # print(original_text)
    # print(trimmed_text)
    assert span_start > -1
    span_end = span_start + len(anaphor_span)

    # first get token start
    token_start = None
    for st in span_and_tokens:
        if len(st) == 0:
            continue
        if st[0] == span_start:
            token_start = st[3]
            break

    # second get token end
    token_end = token_start
    for st in span_and_tokens:
        if len(st) == 0:
            continue
        if st[1] == span_end:
            token_end = st[3]
            break

    # print(token_start, token_end)
    return [token_start, token_end]


def create_mixture_dataset(
    md_outputs: dict,
    md_dataset: dict,
    lm_dataset: dict,
) -> List[dict]:
    """
    dali_outputs: { "0404": { "pred_mentions": [...], "gold_mentions": [...] } }
    original_dataset: { "4040": [[The, mixture,...],[...]] }
    lm_dataset: {"4040-00": ...}
    """
    mixtures = []
    for doc_id, example in md_dataset.items():

        # edge cases doc_id; don't know why disappear; skip for now
        if doc_id in ["1253", "0183"]:
            continue

        # this is to re-align conll tokenizer to character indices
        (span_and_tokens, original_text) = get_spanTokenAlignment_and_originalText(
            doc_id
        )

        # flatten the sentences of this example
        sentences = [t for l in md_dataset[doc_id]["sentences"] for t in l]

        # get total number of predicted anaphors for this doc_key from lm_dataset
        num_anaphors = (
            max(
                [
                    int(example_id.split("-")[1])
                    for example_id in lm_dataset
                    if doc_id in example_id
                ]
            )
            + 1
        )

        # pipeline_anaphor_index. This to make it work with predicted vs. true anaphors work
        p_anaphor_idx = 0
        num_fn = 0  # number of false negatives, to offset
        for i in range(num_anaphors):

            example_id = "%s-0%s" % (doc_id, i) if i < 10 else "%s-%s" % (doc_id, i)

            # ignore false negative cases from anaphor detection
            if example_id not in lm_dataset:
                num_fn += 1

                if doc_id in [
                    "1108",
                    "0773",
                    "0363",
                    "1094",
                    "1269",
                    "1490",
                    "0566",
                ] and example_id not in [
                    "1269-03"
                ]:  # special cases where there's a gap in gold data as well
                    p_anaphor_idx += 1
                continue

            else:
                # false positive cases from anaphor detection
                # BUG here
                if i - num_fn >= len(example["anaphors"]):
                    # TODO need to get the correct tokenized index of this false positive anaphor
                    anaphor_span = lm_dataset[example_id]["anaphor"]
                    anaphor = get_anaphor_tokenIDs_from_text(
                        original_text,
                        lm_dataset[example_id]["context"],
                        anaphor_span,
                        span_and_tokens,
                    )
                else:
                    # [[6,13]] -> [6,13]
                    anaphor = example["anaphors"][p_anaphor_idx][0]
                    p_anaphor_idx += 1
                    # anaphor = example["anaphors"][p_anaphor_idx][0]
                    # p_anaphor_idx += 1
                candidates = find_candidate_antecedents(
                    sentences,
                    anaphor,
                    md_outputs[doc_id]["pred_mentions"],
                    span_and_tokens,
                    original_text,
                )
                anaphor_text = get_original_text_from_conll_tokenized_span(
                    doc_id, anaphor, span_and_tokens, original_text
                )
                # try:
                assert (
                    anaphor_text == lm_dataset[example_id]["anaphor"]
                ), "anaphor mismatched (%s, %s) for example_id=%s" % (
                    anaphor_text,
                    lm_dataset[example_id]["anaphor"],
                    example_id,
                )

                mixtures.append(
                    {
                        "example_id": example_id,
                        "context": lm_dataset[example_id]["context"],
                        "anaphor": anaphor_text,
                        "gold_antecedents": lm_dataset[example_id]["gold_antecedents"],
                        "candidate_antecedents": candidates,
                        "preceding_anaphors": lm_dataset[example_id][
                            "preceding_anaphors"
                        ],
                    }
                )
                # except:
                #     print("Something is wrong with example_id=%s" % example_id)

    return mixtures


def predict_recent_k(dataset, k) -> dict:

    outputs = dict()

    # get the predictions
    for example in dataset:
        # if k == "all" or k > len(example["candidate_antecedents"]):
        #     predictions = example["candidate_antecedents"]
        # else:
        # doing something fancier, where we remove the preceding anaphors
        end_idx = 1
        predictions = []
        for i in range(len(example["candidate_antecedents"]) - 1, -1, -1):
            cur_prediction = example["candidate_antecedents"][i]
            if cur_prediction not in example["preceding_anaphors"]:
                predictions.append(cur_prediction)
                end_idx += 1
            # else:
            #     print(cur_prediction, example["preceding_anaphors"])
            if k != "all" and end_idx > k:
                break
            # predictions = example["candidate_antecedents"][-1 - k + 1 :]
        outputs[example["example_id"]] = {
            "predicted_antecedents": predictions,
            "gold_antecedents": example["gold_antecedents"],
        }

    return outputs


def predict_random(dataset) -> dict:
    outputs = dict()

    # get the predictions
    for example in dataset:
        predictions = []
        for candidate in example["candidate_antecedents"]:
            if random.randint(1, 2) == 1:
                predictions.append(candidate)

        outputs[example["example_id"]] = {
            "predicted_antecedents": predictions,
            "gold_antecedents": example["gold_antecedents"],
        }

    return outputs


def main():

    # eg "logs/biaffinemd_k02_trial0/predictions.json"
    # md_outputs_filepath = sys.argv[1]
    ks = ["k04", "k08", "k16", "k32", "k64"]
    trials = 5
    md_outputs_filepaths, parent_dirs = [], []
    for k in ks:
        for trial in range(trials):
            md_outputs_filepaths.append(
                "../dali-md/logs/biaffinemd_%strial%s/predictions.json" % (k, trial)
            )
            parent_dirs.append(
                "pipeline_baseline_official/%strial%s_baselines_testSmallPredicted"
                % (k, trial)
            )
    md_data_filepath = "/srv/share5/nghia6/data/ChemuRef_v3/pipeline_baseline_data/test_small.jsonlines"
    lm_predicted_data_filepath = "/srv/share5/nghia6/data/ChemuRef_v3/lm_trimmed_data/test_small_predicted_anaphors.jsonl"
    lm_gold_data_filepath = (
        "/srv/share5/nghia6/data/ChemuRef_v3/lm_trimmed_data/test_small.jsonl"
    )
    models = ["recent-2", "recent-3", "recent-4", "recent-5", "recent-all", "random"]
    models = ["recent-5", "recent-all", "random"]

    for (
        md_outputs_filepath,
        parent_dir,
    ) in zip(md_outputs_filepaths, parent_dirs):

        # read dataset
        md_outputs = read_json(md_outputs_filepath)
        md_dataset = dict()
        with open(md_data_filepath, "r") as f:
            for line in f.readlines():
                example = json.loads(line)
                md_dataset[example["doc_key"]] = example
        lm_predicted_dataset = read_jsonl(lm_predicted_data_filepath)
        lm_gold_dataset = read_jsonl(lm_gold_data_filepath)

        # create mixture dataset
        mixture_dataset = create_mixture_dataset(
            md_outputs, md_dataset, lm_predicted_dataset
        )

        # predict and get evaluation metrics for each models
        for model in models:

            if model == "recent-2":
                outputs = predict_recent_k(mixture_dataset, 2)
            elif model == "recent-3":
                outputs = predict_recent_k(mixture_dataset, 3)
            elif model == "recent-4":
                outputs = predict_recent_k(mixture_dataset, 4)
            elif model == "recent-5":
                outputs = predict_recent_k(mixture_dataset, 5)
            elif model == "recent-all":
                outputs = predict_recent_k(mixture_dataset, "all")
            elif model == "random":
                outputs = predict_random(mixture_dataset)

            # compute metrics
            metrics = compute_evaluation_metrics_span_single(
                outputs, lm_gold_dataset, verbose=False
            )

            # print results
            print(
                "F1 score for model=%s/%s: %0.4f"
                % (parent_dir, model, metrics["span_single_f1"])
            )

            # save results and evaluation metrics
            exp_dir = os.path.join(parent_dir, model)
            os.makedirs(exp_dir, exist_ok=True)
            metrics_filepath = os.path.join(exp_dir, "results.json")
            predictions_filepath = os.path.join(exp_dir, "predictions.json")
            for d, filepath in [
                (outputs, predictions_filepath),
                (metrics, metrics_filepath),
            ]:
                with open(filepath, "w") as f:
                    json.dump(d, f, indent=4)


if __name__ == "__main__":
    main()
