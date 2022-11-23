"""Experiment with generating variable number of in-context demonstrations, for a 
single prompt"""
from collections import defaultdict
from typing import List

import os
import sys
import random
import json
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer

from utils import *
from prompt_generation.templates import *

SEED = 30318
random.seed(SEED)


def get_inContext_examples(
    test_example, train_data, candidates, tokenizer, maxInContextNum, max_input_len
) -> List[str]:
    # first encode the test example
    test_len = get_input_length(
        create_chemuref_prompt_without_answer(test_example), tokenizer
    )
    context_len = max_input_len - test_len

    # now try to encode an initial example
    inContextExamples = []
    i = 0
    inContext_len = get_input_length(
        create_chemuref_prompt_with_answers(train_data[candidates[i]]), tokenizer
    )
    while (
        len(inContextExamples) < maxInContextNum  # still fit within maxInContextNum
        and inContext_len < context_len  # can still fit within windows
        and i < len(train_data.keys())
    ):

        # "add" this to the window
        context_len -= inContext_len + 1
        inContextExamples.append(candidates[i])

        # move on to the next case
        if i == len(train_data.keys()) - 1:
            break
        else:
            i += 1
            inContext_len = get_input_length(
                create_chemuref_prompt_with_answers(train_data[candidates[i]]),
                tokenizer,
            )

    return inContextExamples


def random_generator(train_data, test_data, tokenizer, maxInContextNum, max_input_len):

    prompt_map = defaultdict(list)  # keep a 2D list, in case we want to extend
    for test_id, test_example in test_data.items():

        candidates = list(train_data.keys())
        random.shuffle(candidates)
        inContextExamples = get_inContext_examples(
            test_example,
            train_data,
            candidates,
            tokenizer,
            maxInContextNum,
            max_input_len,
        )
        prompt_map[test_id].append(tuple(inContextExamples))

    return prompt_map


def similar_generator(
    train_data,
    test_data,
    similarity_map,
    tokenizer,
    maxInContextNum,
    max_input_len,
    prompt_ordering,
):
    prompt_map = defaultdict(list)
    for test_id, test_example in test_data.items():

        # first gather enough examples
        candidates = list(map(lambda x: x[0], similarity_map[test_id]))
        inContextExamples = get_inContext_examples(
            test_example,
            train_data,
            candidates,
            tokenizer,
            maxInContextNum,
            max_input_len,
        )

        # then do the ordering
        if prompt_ordering == "mostSimilarClosest":
            prompt_map[test_id].append(tuple(reversed(inContextExamples)))

        elif prompt_ordering == "similarMixed":
            random.shuffle(inContextExamples)
            prompt_map[test_id].append(tuple(inContextExamples))

        else:  # default is leastSimilarClosests (eg ex_1 -> ... -> ex_t -> x_test, where ex_t is least similar to x_test)
            prompt_map[test_id].append(tuple(inContextExamples))

    return prompt_map


def main():

    exp_dir = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]
    prompt_ordering = sys.argv[4]
    maxInContextNum = int(sys.argv[5])
    similarity_filepath = os.path.join(exp_dir, "similarity_map.json")
    max_context_len = 2048
    max_generated_len = 256
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # run a bunch of test
    assert prompt_ordering in [
        "random",
        "mostSimilarClosest",
        "leastSimilarClosest",
        "similarMixed",
    ]
    assert 1 <= maxInContextNum <= 8
    assert os.path.exists(similarity_filepath)

    # read in files
    train_data = read_jsonl(train_filepath)
    test_data = read_jsonl(test_filepath)
    similarity_map = read_json(similarity_filepath)

    # now generate prompt based on cases
    if prompt_ordering == "random":
        prompt_map = random_generator(
            train_data,
            test_data,
            tokenizer,
            maxInContextNum,
            max_context_len - max_generated_len,
        )
    else:
        prompt_map = similar_generator(
            train_data,
            test_data,
            similarity_map,
            tokenizer,
            maxInContextNum,
            max_context_len - max_generated_len,
            prompt_ordering,
        )

    # save prompt_map
    write_json(prompt_map, os.path.join(exp_dir, "prompt_map.json"))

    # also generate maxInContextNum statistics and graphs, for good measures
    inContextNums = [len(prompts[0]) for _, prompts in prompt_map.items()]
    # graph
    plt.hist(inContextNums, bins=np.arange(1, max(inContextNums) + 3) - 0.5)
    plt.xlabel("Max Number of In-context Demonstrations")
    plt.xticks(range(1, max(inContextNums) + 3))
    plt.ylabel("Number of Test Examples")
    plt.savefig(os.path.join(exp_dir, "maxInContextNum_distribution.pdf"))
    # metrics: mean, std, median, min, max
    mean = sum(inContextNums) / len(inContextNums)
    variance = sum([((x - mean) ** 2) for x in inContextNums]) / len(inContextNums)
    std = variance**0.5
    metrics = {
        "mean": mean,
        "std": std,
        "median": sorted(inContextNums)[len(inContextNums) // 2],
        "max": max(inContextNums),
        "min": min(inContextNums),
    }
    print(metrics)
    write_json(metrics, os.path.join(exp_dir, "maxInContextNum_metrics.json"))


if __name__ == "__main__":
    main()
