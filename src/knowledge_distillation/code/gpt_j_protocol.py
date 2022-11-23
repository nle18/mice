import os
import json
import sys
import random
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# from utils import read_jsonl

def read_jsonl(filepath: str) -> dict:
    data = dict()
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            data[example["example_id"]] = example
    return data

# from prompt_generation.templates import *
def get_input_length(text, tokenizer) -> int:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_length = input_ids.shape[1]
    return input_length


def create_chemuref_prompt(
    dev_example: dict, train_example1: dict, train_example2: dict
) -> str:

    # add in context examples
    prompt = ""
    for inContext in [train_example1, train_example2]:
        prompt += "Context: %s\nQuestion: What does %s contain?\nAnswer: %s\n\n" % (
            inContext["context"],
            inContext["anaphor"],
            " | ".join(inContext["gold_antecedents"]),
        )

    # add the example itself
    prompt += "Context: %s\nQuestion: What does %s contain?\nAnswer:" % (
        dev_example["context"],
        dev_example["anaphor"],
    )

    return prompt, len(prompt)


def create_chemuref_prompt_without_answer(example: dict) -> str:
    return "Context: %s\nQuestion: What does %s contain?\nAnswer:" % (
        example["context"],
        example["anaphor"],
    )


def create_chemuref_prompt_with_answers(example: dict) -> str:
    return "Context: %s\nQuestion: What does %s contain?\nAnswer: %s\n" % (
        example["context"],
        example["anaphor"],
        " | ".join(example["gold_antecedents"]),
    )

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
    num_prompts,
    model,
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
    for (train_id1, train_id2) in prompts[:num_prompts]:

        if str((train_id1, train_id2)) in predictions:
            continue

        else:

            try:

                input = create_model_input(
                    test_example,
                    train_data[train_id1],
                    train_data[train_id2],
                    tokenizer,
                    max_input_len,
                )
                print("Making predictions for prompt=%s" % str((train_id1, train_id2)))
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
                print("generated_tokens=", generated_tokens)

                # generated text
                generated_text = tokenizer.decode(generated_tokens[0])
                print("generated_text=", generated_text)

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

                print("predictions:", predicted_antecedents)

                predictions[str((train_id1, train_id2))] = {
                    "input_text": input["input_text"],
                    "output_text": generated_text,
                    "predictions": predicted_antecedents,
                    "gold_antecedents": [
                        t.lower() for t in test_example["gold_antecedents"]
                    ],
                }
                logits_info[str((train_id1, train_id2))] = {
                    "prompt_id": (train_id1, train_id2),
                    "output_text": generated_text,
                    "generated_tokens": tokenizer.convert_ids_to_tokens(
                        generated_tokens[0]
                    ),
                    "generated_token_ids": generated_tokens[0].cpu().tolist(),
                    "first_tokens": first_tokens,
                }
                logits[str((train_id1, train_id2))] = first_token_logits
            except:
                print(
                    "%s has some problem, probably exceed max sequence len"
                    % str((train_id1, train_id2))
                )
                predictions[str((train_id1, train_id2))] = {
                    "input_text": input["input_text"],
                    "output_text": [],
                    "predictions": [],
                    "gold_antecedents": [
                        t.lower() for t in test_example["gold_antecedents"]
                    ],
                    "output_probabilities": [],
                }
                logits_info[str((train_id1, train_id2))] = None
                logits[str((train_id1, train_id2))] = np.empty(0)
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


def main():

    exp_dir = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]
    # test_id = sys.argv[4]  # eg "0729-16"
    start_idx = int(sys.argv[4])
    num_prompts = int(sys.argv[5])
    # example_exp_dir = os.path.join(exp_dir, "examples", test_id)
    # os.makedirs(example_exp_dir, exist_ok=True)
    stride=int(sys.argv[6])

    # gpt_id = start_idx//stride
    gpt_id = sys.argv[7]

    os.environ['CUDA_VISIBLE_DEVICES']=str(gpt_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"device: {device}:{gpt_id} | n_gpu: {n_gpu}")

    # read data
    train_data = read_jsonl(train_filepath)
    test_data = read_jsonl(test_filepath)
    # test_example = test_data[test_id]

    print("Load model...", end="")
    # model_type = "/srv/share5/nghia6/codebases/gpt-j-6B"
    model_type = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_type).cuda()
    max_generated_len = 256
    max_context_len = 2048
    print("done!")

    test_id_list = []
    # also get prompt_map file and generate possible in-context configurations
    print("Get prompt_map file...", end="")
    prompt_map_filepath = os.path.join(exp_dir, "prompt_map.json")
    with open(prompt_map_filepath, "r") as f:
        prompt_map = json.load(f)
        test_id_list = sorted(prompt_map.keys())
    print(test_id_list)

    for test_id in test_id_list[start_idx:start_idx+stride]:

        example_exp_dir = os.path.join(exp_dir, test_id)
        print(example_exp_dir)
        os.makedirs(example_exp_dir, exist_ok=True)

        test_example = test_data[test_id]

        # generate predictions
        # first get existing predictions, if exist
        predictions_filepath = os.path.join(example_exp_dir, "predictions.json")
        # logits_info_filepath = os.path.join(example_exp_dir, "logits_info.json")
        # logits_filepath = os.path.join(example_exp_dir, "logits.npz")
        # if os.path.exists(predictions_filepath):
        #     print("Recover predictions, logits_info, logits...", end="")
        #     with open(predictions_filepath, "r") as f:
        #         predictions = json.load(f)
        #     with open(logits_info_filepath, "r") as f:
        #         logits_info = json.load(f)
        #     logits_obj = np.load(logits_filepath)
        #     logits = dict()
        #     for k in predictions:
        #         logits[k] = logits_obj[k]
        #     print("done!")
        # else:
        predictions = dict()
        logits_info = dict()
        logits = dict()

        # # also get prompt_map file and generate possible in-context configurations
        # print("Get prompt_map file...", end="")
        # prompt_map_filepath = os.path.join(exp_dir, "prompt_map.json")
        # with open(prompt_map_filepath, "r") as f:
        #     prompt_map = json.load(f)
        # print("done!")

        # make predictions
        end_prediction = False
        while not end_prediction:
            predictions, logits_info, logits, end_prediction = predict_with_gptj(
                test_example,
                train_data,
                prompt_map[test_id],
                num_prompts,
                model,
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
            # with open(logits_info_filepath, "w") as f:
            #     json.dump(logits_info, f, indent=4)
            # np.savez_compressed(logits_filepath, **logits)
            monitor_filepath = os.path.join(example_exp_dir, "monitoring.json")
            with open(monitor_filepath, "w") as f:
                json.dump({"num_predictions": len(predictions.keys())}, f, indent=4)
            print("done!")


if __name__ == "__main__":
    main()
