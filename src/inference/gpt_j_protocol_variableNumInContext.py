import os
import json
import sys
import random
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp

from utils import read_jsonl, read_txt
from prompt_generation.templates import *


def create_model_input(
    test_example, train_data, inContextExamples, tokenizer, max_input_len, device
):

    # add all the in-context examples
    input_text = ""
    for train_id in inContextExamples:

        # construct the final input_ids
        input_text += "%s\n" % create_chemuref_prompt_with_answers(train_data[train_id])

    # finally add test examples
    input_text += create_chemuref_prompt_without_answer(test_example)

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    return {"input_ids": input_ids, "input_text": input_text}


def predict_with_gptj(
    test_example,
    train_data,
    prompts,
    num_prompts,
    model,
    device,
    tokenizer,
    max_context_len,
    max_generated_len,
    predictions,
    logits_info,
    logits,
) -> dict:

    max_input_len = 2048 - max_generated_len

    # make predictions
    prev_num_predictions = len(predictions.keys())
    cur_num_predictions = prev_num_predictions
    for inContextExamples in prompts[:num_prompts]:

        if str(inContextExamples) in predictions:
            continue

        else:

            try:

                input = create_model_input(
                    test_example,
                    train_data,
                    inContextExamples,
                    tokenizer,
                    max_input_len,
                    device,
                )
                print(
                    "GPU=%s: Making predictions for prompt=%s"
                    % (device, str(inContextExamples))
                )
                input_len = input["input_ids"].shape[1]
                outputs = model.generate(
                    input["input_ids"],
                    max_new_tokens=max_generated_len,
                    temperature=0,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=198,  # special character 'ċ' (bytecode for new line?) NOTE use this for generation
                )
                # generated tokens
                generated_tokens = outputs.sequences[:, input_len:-1]
                generated_len = generated_tokens.shape[1]
                # print("generated_tokens=", generated_tokens)

                # generated text
                generated_text = tokenizer.decode(generated_tokens[0])
                # print("generated_text=", generated_text)

                # get the first token probabilities
                first_tokens = []
                first_token_logits = []
                sep_id = 930  # " |"
                for i in range(generated_len):

                    # add first token
                    if i == 0:
                        cur_token_id = generated_tokens[0, i].item()
                        token_logits = outputs.scores[i]
                        first_token_logits.append(token_logits)
                        first_tokens.append(
                            (cur_token_id, tokenizer.decode([cur_token_id]))
                        )

                    else:
                        cur_token_id = generated_tokens[0, i].item()
                        # if current is SEP token, then add next token
                        if cur_token_id == sep_id and i + 1 < generated_len - 1:
                            next_token_id = generated_tokens[0, i + 1].item()
                            token_logits = outputs.scores[i + 1]
                            first_token_logits.append(token_logits)
                            first_tokens.append(
                                (next_token_id, tokenizer.decode([next_token_id]))
                            )
                first_token_logits = torch.cat(first_token_logits, dim=0).cpu().numpy()

                # post-process strings: filter out all empty strings and duplications
                predicted_antecedents = [
                    p.strip().lower() for p in generated_text.strip().split("|")
                ]
                predicted_antecedents = list(
                    filter(lambda a: a != "", predicted_antecedents)
                )
                predicted_antecedents = list(set(predicted_antecedents))

                # print("predictions:", predicted_antecedents)

                predictions[str(inContextExamples)] = {
                    "input_text": input["input_text"],
                    "output_text": generated_text,
                    "predictions": predicted_antecedents,
                    "gold_antecedents": [
                        t.lower() for t in test_example["gold_antecedents"]
                    ],
                }
                logits_info[str(inContextExamples)] = {
                    "prompt_id": inContextExamples,
                    "output_text": generated_text,
                    "generated_tokens": tokenizer.convert_ids_to_tokens(
                        generated_tokens[0]
                    ),
                    "generated_token_ids": generated_tokens[0].cpu().tolist(),
                    "first_tokens": first_tokens,
                }
                logits[str(inContextExamples)] = first_token_logits
            except:
                print(
                    "%s has some problem, probably exceed max sequence len"
                    % str(inContextExamples)
                )
                predictions[str(inContextExamples)] = {
                    "input_text": input["input_text"],
                    "output_text": [],
                    "predictions": [],
                    "gold_antecedents": [
                        t.lower() for t in test_example["gold_antecedents"]
                    ],
                    "output_probabilities": [],
                }
                logits_info[str(inContextExamples)] = None
                logits[str(inContextExamples)] = np.empty(0)
            cur_num_predictions += 1

            # save every 10 examples
            if cur_num_predictions % 10 == 0:
                break

    return (
        predictions,
        logits_info,
        logits,
        cur_num_predictions == prev_num_predictions,
    )


def run_inference(
    rank, exp_dir, train_data, test_data, num_prompts, num_gpus, starting_gpu
):

    gpu_id = starting_gpu + rank
    device = torch.device("cuda:%s" % gpu_id if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | num_gpu: {num_gpus}")

    print("Load model...", end="")
    model_type = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_type).to(device)
    max_generated_len = 256
    max_context_len = 2048
    print("done!")

    all_examples = read_txt(os.path.join(exp_dir, "test_examples.txt"))
    num_examples_per_gpu = len(all_examples) // num_gpus
    start = rank * num_examples_per_gpu
    end = (
        (rank + 1) * num_examples_per_gpu if rank < num_gpus - 1 else len(all_examples)
    )
    examples = all_examples[start:end]
    print("Load %s examples for gpu_id=%s" % (len(examples), gpu_id))
    print("done!")

    for test_id in examples:
        example_exp_dir = os.path.join(exp_dir, "examples", test_id)
        os.makedirs(example_exp_dir, exist_ok=True)
        test_example = test_data[test_id]

        # generate predictions
        # first get existing predictions, if exist
        predictions_filepath = os.path.join(example_exp_dir, "predictions.json")
        logits_info_filepath = os.path.join(example_exp_dir, "logits_info.json")
        logits_filepath = os.path.join(example_exp_dir, "logits.npz")
        try:
            if os.path.exists(predictions_filepath):
                print(
                    "Recover predictions, logits_info, logits for example=%s..."
                    % test_id,
                    end="",
                )
                with open(predictions_filepath, "r") as f:
                    predictions = json.load(f)
                with open(logits_info_filepath, "r") as f:
                    logits_info = json.load(f)
                logits_obj = np.load(logits_filepath)
                logits = dict()
                for k in predictions:
                    logits[k] = logits_obj[k]
                print("done!")
            else:
                predictions = dict()
                logits_info = dict()
                logits = dict()
        except:
            print("cannot recover; run inference on this example again")
            predictions = dict()
            logits_info = dict()
            logits = dict()

        # also get prompt_map file and generate possible in-context configurations
        print("Get prompt_map file for example_id=%s..." % test_id, end="")
        prompt_map_filepath = os.path.join(exp_dir, "prompt_map.json")
        with open(prompt_map_filepath, "r") as f:
            prompt_map = json.load(f)
        print("done!")

        # make predictions
        end_prediction = False
        while not end_prediction:
            predictions, logits_info, logits, end_prediction = predict_with_gptj(
                test_example,
                train_data,
                prompt_map[test_id],
                num_prompts,
                model,
                device,
                tokenizer,
                max_context_len,
                max_generated_len,
                predictions,
                logits_info,
                logits,
            )

            print("Save %s predictions..." % len(predictions.keys()), end="")

            # save outputs and monitor information
            with open(predictions_filepath, "w") as f:
                json.dump(predictions, f, indent=4)
            with open(logits_info_filepath, "w") as f:
                json.dump(logits_info, f, indent=4)
            np.savez_compressed(logits_filepath, **logits)
            monitor_filepath = os.path.join(example_exp_dir, "monitoring.json")
            with open(monitor_filepath, "w") as f:
                json.dump({"num_predictions": len(predictions.keys())}, f, indent=4)
            print("done!")


def main():

    exp_dir = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]
    num_prompts = int(sys.argv[4])
    num_gpus = int(sys.argv[5])
    starting_gpu = int(sys.argv[6])

    # read data
    train_data = read_jsonl(train_filepath)
    test_data = read_jsonl(test_filepath)

    mp.spawn(
        run_inference,
        args=(exp_dir, train_data, test_data, num_prompts, num_gpus, starting_gpu),
        nprocs=num_gpus,
    )


if __name__ == "__main__":
    main()
