import os
import json
import sys
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *
from prompt_generation.templates import *
from calibration.compute_calibration_parameters import *


def create_model_input(
    test_example, train_example1, train_example2, tokenizer, max_input_len
):
    """Employ the truncating tactic of truncating the test context"""
    test_len = get_input_length(
        create_chemuref_prompt_without_answer(test_example), tokenizer
    )

    # get context len of ex1 and ex2
    ex1_len = get_input_length(
        create_chemuref_prompt_with_answers(train_example1), tokenizer
    )
    ex2_len = get_input_length(
        create_chemuref_prompt_with_answers(train_example2), tokenizer
    )

    # get encoded for test_example
    test_text = create_chemuref_prompt_without_answer(test_example)
    test_input_ids = tokenizer.encode(test_text, return_tensors="pt")
    test_len = test_input_ids.shape[1]

    # truncated them accordingly
    test_remaining_len = (
        max_input_len - ex1_len - ex2_len - 3
    )  # 3 is just some buffering amount
    test_input_ids = test_input_ids[0, test_len - test_remaining_len :]

    # map back to string to construct prompt
    truncated_test_text = tokenizer.decode(test_input_ids)
    if not truncated_test_text.startswith("Context: "):
        truncated_test_text = "Context: " + truncated_test_text

    # construct the final input_ids
    input_text = "%s\n%s\n%s" % (
        create_chemuref_prompt_with_answers(train_example1),
        create_chemuref_prompt_with_answers(train_example2),
        truncated_test_text,
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()

    return {"input_ids": input_ids, "input_text": input_text}


def predict_with_gptj(
    test_example,
    train_data,
    prompts,
    model,
    tokenizer,
    max_context_len,
    max_generated_len,
) -> dict:

    max_input_len = 2048 - max_generated_len
    (train_id1, train_id2) = prompts[0]

    # make predictions
    predictions = dict()
    # try:
    print("Making predictions for prompt=%s" % str((train_id1, train_id2)))
    # first compute (uncalibrated) logits of first token
    input = create_model_input(
        test_example,
        train_data[train_id1],
        train_data[train_id2],
        tokenizer,
        max_input_len,
    )
    input_len = input["input_ids"].shape[1]
    outputs = model.generate(
        input["input_ids"],
        max_new_tokens=max_generated_len,
        return_dict_in_generate=True,
        output_scores=True,
        eos_token_id=198,
    )

    # generated tokens; input_len - 1 accounted for calibrated first token
    generated_tokens = outputs.sequences[:, input_len:-1]
    generated_len = generated_tokens.shape[1]
    # print("generated_tokens=", generated_tokens)

    # generated text
    generated_text = tokenizer.decode(generated_tokens[0])
    print("generated_text=", generated_text)

    # post-process strings: filter out all empty strings and duplications
    predicted_antecedents = [
        p.strip().lower() for p in generated_text.strip().split("|")
    ]
    predicted_antecedents = list(filter(lambda a: a != "", predicted_antecedents))
    predicted_antecedents = list(set(predicted_antecedents))

    print("predictions:", predicted_antecedents)

    predictions[str((train_id1, train_id2))] = {
        "input_text": input["input_text"],
        "output_text": generated_text,
        "predictions": predicted_antecedents,
        "gold_antecedents": [t.lower() for t in test_example["gold_antecedents"]],
    }
    # except:
    #     print(
    #         "%s has some problem, probably exceed max sequence len"
    #         % str((train_id1, train_id2))
    #     )
    #     predictions[str((train_id1, train_id2))] = {
    #         "input_text": input["input_text"],
    #         "output_text": [],
    #         "predictions": [],
    #         "gold_antecedents": [t.lower() for t in test_example["gold_antecedents"]],
    #     }

    return predictions


def main():

    exp_dir = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]

    # read data
    train_data = read_jsonl(train_filepath)
    test_data = read_jsonl(test_filepath)

    print("Load prompt_map...", end="")
    prompt_map_filepath = os.path.join(exp_dir, "prompt_map.json")
    with open(prompt_map_filepath, "r") as f:
        prompt_map = json.load(f)
    print("done!")

    print("Load model...", end="")
    model_type = "/srv/share5/nghia6/codebases/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_type).cuda()
    max_generated_len = 256
    max_context_len = 2048
    print("done!")

    # now make predictions
    for test_id, test_example in test_data.items():

        # check if has predictions, if not, then make predictions
        example_folderpath = os.path.join(exp_dir, "examples", test_id)
        predictions_filepath = os.path.join(example_folderpath, "predictions.json")
        if not os.path.exists(predictions_filepath):
            os.makedirs(example_folderpath, exist_ok=True)

            # make predictions
            predictions = predict_with_gptj(
                test_example,
                train_data,
                prompt_map[test_id],
                model,
                tokenizer,
                max_context_len,
                max_generated_len,
            )

            # save predictions
            write_json(predictions, predictions_filepath)


if __name__ == "__main__":
    main()
