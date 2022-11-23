"""Code for finetuning gpt2 model
Code taken and adapted from: 
- http://mohitmayank.com/a_lazy_data_science_guide/natural_language_processing/GPTs/#introduction
- https://towardsdatascience.com/guide-to-fine-tuning-text-generation-models-gpt-2-gpt-neo-and-t5-dc5de6b3bc5e

NOTE: 
- We can either finetune with or without in-context demonstrations. For simplicity 
sake, let's try without (03/17). For with in-context, we can follow LM-BFF (Gao et al)
"""
from typing import List
from difflib import SequenceMatcher

import sys
import os
import torch
import json

from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

BOS_TOKEN = "<|startoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
SEP_TOKEN = "<SEP>"  # NOTE: previously we use " | " for separator


def compute_true_positives(
    gold_antecedents: List[str], predicted_antecedents: List[str]
):

    tp = 0
    similarity_threshold = 1.0  # range is 0.0 - 1.0 # NOTE: experiment with the ranges
    for predicted_antecedent in predicted_antecedents:
        for gold_antecedent in gold_antecedents:

            #  How to compare the strings?
            if (
                SequenceMatcher(None, predicted_antecedent, gold_antecedent).ratio()
                >= similarity_threshold
            ):
                tp += 1

    return tp


def compute_evaluation_metrics(outputs: dict) -> dict:

    gold_num_antecedents = 0
    predicted_num_antecedents = 0
    tp = 0
    num_examples = 0
    num_correct = 0
    for example_id, example in outputs.items():

        gold_antecedents = example["gold_antecedents"]
        predicted_antecedents = example["predictions"]

        cur_tp = compute_true_positives(gold_antecedents, predicted_antecedents)
        tp += cur_tp
        gold_num_antecedents += len(gold_antecedents)
        predicted_num_antecedents += len(predicted_antecedents)
        num_examples += 1

        if cur_tp == len(gold_antecedents) == len(predicted_antecedents):
            num_correct += 1

    p = tp / predicted_num_antecedents
    r = tp / gold_num_antecedents
    return {
        "lenient_precision": p,
        "lenient_recall": r,
        "lenient_f1": 2 * p * r / (p + r) if (p + r) != 0 else 0,
        "strict_accuracy": num_correct / num_examples,
    }


def read_dataset(filepath: str, tokenizer, max_context_len, train=True) -> List[dict]:
    """Return a list of tuple (input_ids, attention_mask, gold_antecedents)"""
    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)

            # get stuff NOTE this is where we create prompt-- can modularize this
            prompt = ""
            if train:
                prompt = (
                    "Context: %s\n Question: What does %s contain?\nAnswer: %s\n\n"
                    % (
                        example["context"],
                        example["anaphor"],
                        SEP_TOKEN.join(example["gold_antecedents"]),
                    )
                )
                encodings_dict = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_context_len,
                    padding="max_length",
                )
                input_ids = torch.tensor(encodings_dict["input_ids"]).cuda()
                attention_mask = torch.tensor(encodings_dict["attention_mask"]).cuda()
            else:
                prompt = "Context: %s\n Question: What does %s contain?\nAnswer:" % (
                    example["context"],
                    example["anaphor"],
                )
                input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
                attention_mask = None
            data.append(
                {
                    "example_id": example["example_id"],
                    "input_text": prompt,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "gold_antecedents": example["gold_antecedents"],
                    "prediction_start": len(prompt),
                }
            )

    return data


def main():

    # load parameters
    # train_filepath = "/srv/share5/nghia6/data/ChEMU-Ref/mixture/few_shot_data_for_lms/train_data/k02/trial4.jsonl"
    # train_filepath = "/srv/share5/nghia6/data/ChEMU-Ref/mixture/few_shot_data_for_lms/train_data/kfull/trial0.jsonl"
    train_filepath = "/srv/share5/nghia6/data/ChEMU-Ref/mixture/few_shot_data_for_lms/train_data/k16/trial4.jsonl"
    dev_filepath = (
        "/srv/share5/nghia6/data/ChEMU-Ref/mixture/few_shot_data_for_lms/dev.jsonl"
    )
    exp_dir = "./gpt2xl_k16/finetuning"  # "./gpt2xl_k02/finetuning"
    os.makedirs(exp_dir, exist_ok=True)
    num_epochs = 5
    max_context_len = 512  # or 1024
    max_generated_len = 512

    ## Load model and data
    # --------

    # set model name and seed
    model_name = "gpt2-xl"
    torch.manual_seed(42)

    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
    )
    model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    model.resize_token_embeddings(len(tokenizer))

    # prepare and load dataset
    train_dataset = read_dataset(train_filepath, tokenizer, max_context_len, train=True)
    test_dataset = read_dataset(dev_filepath, tokenizer, max_context_len, train=False)

    ## Train
    # --------
    # creating training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, "results"),
        num_train_epochs=num_epochs,
        logging_steps=10,  # NOTE: what is this, is this necessary?
        # load_best_model_at_end=True,
        # save_strategy="epoch",
        # evaluation_strategy="epoch", # TODO: add eval set if use eval
        evaluation_strategy="no",
        per_device_train_batch_size=1,  # NOTE: this may introduce bug
        # per_device_eval_batch_size=1,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(exp_dir, "logs"),
        dataloader_pin_memory=False,
    )

    # start training
    print("Start training...")
    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["input_ids"] for f in data]),
        },
    ).train()
    print("Finish training")

    ## Test
    # ----------

    # set the model to eval mode
    _ = model.eval()

    # run model inference on all test data
    predictions = dict()
    # iter over all of the test data
    for i, test_example in enumerate(test_dataset):

        # perform prediction
        sample_outputs = model.generate(
            test_example["input_ids"].cuda(),
            do_sample=False,
            top_k=50,
            max_length=max_context_len,
            top_p=0.90,
            temperature=0,
            num_return_sequences=0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # decode the predicted tokens into texts
        output_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

        # extract predictions (and other information) according to our scheme
        generated_text = output_text[test_example["prediction_start"] :]
        predictions_str = generated_text[: generated_text.find(EOS_TOKEN)]

        # post-process strings: filter out all empty strings and duplications
        predicted_antecedents = [
            p.strip().lower() for p in predictions_str.strip().split(SEP_TOKEN)
        ]
        predicted_antecedents = list(filter(lambda a: a != "", predicted_antecedents))
        predicted_antecedents = list(set(predicted_antecedents))

        predictions[test_example["example_id"]] = {
            "input_text": test_example["input_text"],
            "output_text": predictions_str,
            "predictions": predicted_antecedents,
            "gold_antecedents": [t.lower() for t in test_example["gold_antecedents"]],
        }

        # save predictions every 50 samples TODO
        if i % 50 == 0:
            print("Save predictions...", end="")

            # compute metrics
            metrics = compute_evaluation_metrics(predictions)
            metrics["num_predictions"] = len(predictions.keys())

            # save outputs
            metrics_filepath = os.path.join(exp_dir, "metrics.json")
            predictions_filepath = os.path.join(exp_dir, "predictions.json")
            with open(predictions_filepath, "w") as f:
                json.dump(predictions, f, indent=4)
            with open(metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=4)
            print("done!")

    # final compute metrics and save outputs
    metrics = compute_evaluation_metrics(predictions)
    metrics["num_predictions"] = len(predictions.keys())

    # save outputs
    metrics_filepath = os.path.join(exp_dir, "metrics.json")
    predictions_filepath = os.path.join(exp_dir, "predictions.json")
    with open(predictions_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()