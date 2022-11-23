"""KATE + Nucleus Sampling experiments (10/18/2022)
1 prompt, but generate 256 independent sequences 
"""
import os
import json
import sys
import random
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import read_jsonl
from prompt_generation.templates import *


def create_model_input(
    test_example, train_data, inContextExamples, tokenizer, max_input_len
):

    # add all the in-context examples
    input_text = ""
    for train_id in inContextExamples:

        # construct the final input_ids
        input_text += "%s\n" % create_chemuref_prompt_with_answers(train_data[train_id])

    # finally add test examples
    input_text += create_chemuref_prompt_without_answer(test_example)

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
    predictions,
    num_generated_sequence=256
) -> dict:

    max_input_len = 2048 - max_generated_len

    # make predictions TODO: sample num_generated_sequence - len(predictions)
    num_predictions = len(predictions.keys())
    inContextExamples = prompts[0]

    input = create_model_input(
        test_example,
        train_data,
        inContextExamples,
        tokenizer,
        max_input_len,
    )
    input_len = input["input_ids"].shape[1]
    
    for i in range(num_predictions, num_generated_sequence):
    
        print("Prediction of sequence {0}".format(i))

        outputs = model.generate(
            input["input_ids"],
            max_new_tokens=max_generated_len,
            do_sample=True,
            temperature=1.0,
            top_k=50, 
            top_p=0.95,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=198,  # special character 'ċ' (bytecode for new line?) NOTE use this for generation
        )
        
        # generated tokens
        generated_tokens = outputs.sequences[:, input_len:-1]
        generated_len = generated_tokens.shape[1]
        print("generated_tokens=", generated_tokens)

        # generated text
        generated_text = tokenizer.decode(generated_tokens[0])
        print("generated_text=", generated_text.strip())

        # post-process strings: filter out all empty strings and duplications
        predicted_antecedents = [
            p.strip().lower() for p in generated_text.strip().split("|")
        ]
        predicted_antecedents = list(
            filter(lambda a: a != "", predicted_antecedents)
        )
        predicted_antecedents = list(set(predicted_antecedents))

        print("predictions:", predicted_antecedents)

        predictions["sequence_{0}".format(i)] = {
            "input_text": input["input_text"],
            "output_text": generated_text,
            "predictions": predicted_antecedents,
            "gold_antecedents": [
                t.lower() for t in test_example["gold_antecedents"]
            ],
        }
    return predictions


def main():

    exp_dir = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]
    test_id = sys.argv[4]  # eg "0729-16"
    example_exp_dir = os.path.join(exp_dir, "examples", test_id)
    os.makedirs(example_exp_dir, exist_ok=True)

    # read data
    train_data = read_jsonl(train_filepath)
    test_data = read_jsonl(test_filepath)
    test_example = test_data[test_id]

    print("Load model...", end="")
    model_type = "/srv/share5/nghia6/codebases/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_type).cuda()
    max_generated_len = 256
    max_context_len = 2048
    print("done!")

    # generate predictions
    # first get existing predictions, if exist
    predictions_filepath = os.path.join(example_exp_dir, "predictions.json")
    if os.path.exists(predictions_filepath):
        print("Recover predictions...", end="")
        with open(predictions_filepath, "r") as f:
            predictions = json.load(f)
        print("done!")
    else:
        predictions = dict()

    # also get prompt_map file and generate possible in-context configurations
    print("Get prompt_map file...", end="")
    prompt_map_filepath = os.path.join(exp_dir, "prompt_map.json")
    with open(prompt_map_filepath, "r") as f:
        prompt_map = json.load(f)
    print("done!")

    # make predictions
    predictions = predict_with_gptj(
        test_example,
        train_data,
        prompt_map[test_id],
        model,
        tokenizer,
        max_context_len,
        max_generated_len,
        predictions,
    )

    print("Save %s predictions..." % len(predictions.keys()), end="")

    # save outputs and monitor information
    with open(predictions_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
    monitor_filepath = os.path.join(example_exp_dir, "monitoring.json")
    with open(monitor_filepath, "w") as f:
        json.dump({"num_predictions": len(predictions.keys())}, f, indent=4)
    print("done!")


if __name__ == "__main__":
    main()
