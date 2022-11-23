"""
Inputs: 
- See main

Outputs: 
- prompt_map.json: for each test example, a list of (example1, example2) (num_prompts pairs)
- similarity_map.json (for "similar" only): for each test example, a list of 
    (example, score) (sqrt(num_prompts) pairs)

TODO:
- Right now only support Random + Roberta tokenizers. If we want to run experiments, 
we can add others, like BM25, ProcBERT, or SentenceT5
"""

from collections import defaultdict
from math import sqrt
import os
import sys
import random
import json
import torch
import torch.nn as nn

from transformers import AutoTokenizer
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer

from utils import read_jsonl
from prompt_generation.templates import *

SEED = 30318
random.seed(SEED)


def random_generator(train_data, test_data, num_prompts):
    prompt_map = defaultdict(list)
    # randomly sample num_prompts pairs from k*k pairs
    all_examples = list(train_data.keys())
    all_pairs = [(ex1, ex2) for ex1 in all_examples for ex2 in all_examples]
    for test_id, test_example in test_data.items():

        # randomly sample num_prompts pairs
        # with (random.choices) or without replacement (random.sample)
        # Doesn't really matter -- results are distinct anyway (because I saved them in a json file)
        # prompt_map[test_id] = random.choices(all_pairs, k=num_prompts)
        prompt_map[test_id] = random.sample(all_pairs, k=num_prompts)

    return prompt_map


def similar_generator(train_data, test_data, num_prompts, ordering=0) -> dict:

    # similarity function and encoder
    model_name = "all-roberta-large-v1"
    model = SentenceTransformer(model_name)
    device = torch.device("cuda")
    model.to(device)

    # first get all the pair-wise similarity scores
    similarity_map = defaultdict(list)
    for test_id, test_example in test_data.items():

        # get test encoding
        prompted_x_test = create_chemuref_prompt_without_answer(test_example)
        x_test_embedding = torch.tensor(model.encode(prompted_x_test))
        for train_id, train_example in train_data.items():

            # get train encoding
            prompted_x_train = create_chemuref_prompt_without_answer(train_example)
            x_train_embedding = torch.tensor(model.encode(prompted_x_train))

            # get similarity score; squeeze to mainly serve eucliean distance
            score = nn.CosineSimilarity(dim=1)(
                x_test_embedding.unsqueeze(0),
                x_train_embedding.unsqueeze(0),
            ).item()
            similarity_map[test_id].append((train_id, score))

    # then create prompts map
    prompt_map = dict()
    for test_id, test_example in test_data.items():

        # first sort all the data in all_map
        similarity_map[test_id] = sorted(
            similarity_map[test_id], key=lambda x: x[1], reverse=True
        )

        # then extract the top-sqrt(num_prompts) from similarity and generate all pairs
        t = int(sqrt(num_prompts))

        # special case for t=1: select the top-2 similar and put into a single prompt
        if t == 1:
            top_2 = list(map(lambda x: x[0], similarity_map[test_id][:2]))
            prompt_map[test_id] = [(top_2[0], top_2[1])]
        else:
            top_t = list(map(lambda x: x[0], similarity_map[test_id][:t]))
            prompt_map[test_id] = [(ex1, ex2) for ex1 in top_t for ex2 in top_t]

    return prompt_map, similarity_map


def main():

    exp_dir = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]
    generation = sys.argv[4]
    num_prompts = int(sys.argv[5])
    encoder = sys.argv[6]
    assert generation in ["random", "similar"]
    assert encoder in ["roberta", "bm25", "procbert", "sentenceT5"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    max_context_len = 2048
    max_generated_len = 256
    os.makedirs(exp_dir, exist_ok=True)

    # read datasets
    train_data = read_jsonl(train_filepath)
    test_data = read_jsonl(test_filepath)

    # first do random prompt generation
    if generation == "random":
        prompt_map = random_generator(train_data, test_data, num_prompts)
    # similar case
    elif generation == "similar":
        prompt_map, similarity_map = similar_generator(
            train_data, test_data, num_prompts
        )

        # save similarity_map
        similarity_map_filepath = os.path.join(exp_dir, "similarity_map.json")
        with open(similarity_map_filepath, "w") as f:
            json.dump(similarity_map, f, indent=4)

    # save prompt_map
    prompt_map_filepath = os.path.join(exp_dir, "prompt_map.json")
    with open(prompt_map_filepath, "w") as f:
        json.dump(prompt_map, f, indent=4)


if __name__ == "__main__":
    main()
