import json
import argparse
import torch
import sys

from typing import List
from difflib import SequenceMatcher

import numpy as np

from tqdm import tqdm, trange
from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import pipeline, set_seed
from collections import Counter

SEP_TOKEN = "<sep>"
EOS_TOKEN = "<|endoftext|>"

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)


def generate_data_full(data_item):
    return f"Context: {data_item['context']}\n Question: What does {data_item['anaphor']} contain?\nAnswer: {SEP_TOKEN.join(data_item['gold_antecedents'])}{EOS_TOKEN}"

def generate_data_prompt(data_item):
    return f"Context: {data_item['context']}\n Question: What does {data_item['anaphor']} contain?\nAnswer: "

def load_data(data_path, input_tokenizer, args, data_type="train"):

    data_list = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    data_full_list = []
    data_prompt_list = []
    # print(data_list[0])

    for data_item in data_list:
        data_full = generate_data_full(data_item)
        data_prompt = generate_data_prompt(data_item)

        data_full_list.append(data_full)
        data_prompt_list.append(data_prompt)

    data_full = input_tokenizer(data_full_list, padding=True, max_length=args.max_length, truncation=True, return_tensors='pt')
    data_prompt = input_tokenizer(data_prompt_list, padding=True, max_length=args.max_length, truncation=True, return_tensors='pt')

    full_input_ids, full_att_mask, full_labels, prompt_input_ids, prompt_att_mask = [], [], [], [], []

    for full_id_item, full_mask_item, prompt_id_item, prompt_mask_item in zip(data_full.input_ids, data_full.attention_mask, data_prompt.input_ids, data_prompt.attention_mask):
        
        prompt_len = torch.sum(prompt_mask_item).item()
        if prompt_len == args.max_length:
            continue

        full_label_item = full_id_item.clone()
        full_label_item[:prompt_len] = -100
        full_label_item[full_label_item == input_tokenizer.pad_token_id] = -100

        full_input_ids.append(full_id_item)
        full_att_mask.append(full_mask_item)
        full_labels.append(full_label_item)
        prompt_input_ids.append(prompt_id_item)
        prompt_att_mask.append(prompt_mask_item)
   
    full_input_ids = torch.stack(full_input_ids, dim=0)
    full_att_mask = torch.stack(full_att_mask, dim=0)
    full_labels = torch.stack(full_labels, dim=0)
    prompt_input_ids = torch.stack(prompt_input_ids, dim=0)
    prompt_att_mask = torch.stack(prompt_att_mask, dim=0)

    dataset = TensorDataset(full_input_ids, full_att_mask, full_labels, prompt_input_ids, prompt_att_mask)
    if data_type == "train":
        data_sampler = RandomSampler(dataset)
    else:
        data_sampler = SequentialSampler(dataset)

    if data_type == "train":
        data_dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=1)
    else:
        data_dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=4)

    return data_dataloader


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


def main(args):
    
    # Set seed
    set_seed(args)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer.add_special_tokens({'sep_token': '<sep>'})
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("Loading data...\n")
    train_path = f"{args.data_dir}/train/k{args.n_shots:02}/trial0.jsonl"
    print(train_path)
    train_dataloader = load_data(train_path, tokenizer, args, data_type="train")

    dev_path = f"{args.data_dir}/dev.jsonl"
    print(dev_path)
    dev_dataloader = load_data(dev_path, tokenizer, args, data_type="dev")

    test_path = f"{args.data_dir}/test_small.jsonl"
    print(test_path)
    test_dataloader = load_data(test_path, tokenizer, args, data_type="test")

    # Load model
    print("Loading model...\n")
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.pad_token_id)
    model.resize_token_embeddings(len(tokenizer))

    # Config the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if n_gpu > 1:
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_f1 = -1

    # Start training
    print("Start training...\n")
    for num_epoch in trange(args.n_epochs, desc="Epoch"):

        # Training
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        with tqdm(total=len(train_dataloader), file=sys.stdout) as pbar:
            for step, batch in list(enumerate(train_dataloader)):

                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_full_input_ids, b_full_input_mask, b_full_lables, _, _ = batch

                # forward pass
                outputs = model(input_ids=b_full_input_ids, attention_mask=b_full_input_mask, labels=b_full_lables)
                loss = outputs.loss

                if n_gpu > 1:
                    loss = loss.mean()

                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
                # update parameters
                optimizer.step()
                model.zero_grad()
                pbar.update(1)

        print(f"Epoch: {num_epoch}")
        print(f"Train loss: {tr_loss / nb_tr_steps}")

        # Validation
        model.eval()

        # Tracking variables
        eval_correct, eval_total = 0, 0
        predictions = {}
        pred_num = 0
        gold_sen_list = []
        pred_sen_list = []
        with tqdm(total=len(dev_dataloader), file=sys.stdout) as pbar:
            for step, batch in enumerate(dev_dataloader):

                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_full_input_ids, b_full_input_mask, _, b_prompt_input_ids, b_prompt_input_mask = batch
                # print(b_input_ids.shape, b_input_mask.shape, b_labels.shape)

                # forward pass
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=b_prompt_input_ids, 
                        attention_mask=b_prompt_input_mask, 
                        top_k=50,
                        top_p=0.9,
                        do_sample=False,
                        max_length=args.max_length, 
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                    gold_sens = tokenizer.batch_decode(b_full_input_ids, skip_special_tokens=True)
                    pred_sens = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

                    gold_sen_list += gold_sens
                    pred_sen_list += pred_sens
                    # print(len(gold_sens), gold_sens[0])
                    # print(len(pred_sens), pred_sens[0])

                for gold_sen, pred_sen in zip(gold_sens, pred_sens):
                    # gold_sen = gold_sens[0]
                    gold_str = gold_sen.split("Answer: ")[1]
                    gold_antecedents = [
                        p.strip().lower() for p in gold_str.strip().split(SEP_TOKEN)
                    ]

                    # pred_sen = pred_sens[0]
                    predictions_str = pred_sen.split("Answer: ")[1]
                    predicted_antecedents = [
                        p.strip().lower() for p in predictions_str.strip().split(SEP_TOKEN)
                    ]
                    predicted_antecedents = list(filter(lambda a: a != "", predicted_antecedents))
                    predicted_antecedents = list(set(predicted_antecedents))

                    # print(gold_antecedents)
                    # print(predicted_antecedents)

                    predictions[pred_num] = {"gold_antecedents": gold_antecedents, "predictions": predicted_antecedents}
                    pred_num += 1

                    # break
                pbar.update(1)

        # final compute metrics and save outputs
        metrics = compute_evaluation_metrics(predictions)
        metrics["num_predictions"] = len(predictions.keys())

        if metrics["lenient_f1"] > best_f1:
            best_f1 = metrics["lenient_f1"]
            print(f"Dev lenient F1: {metrics['lenient_f1']}")
            print(metrics)
        
            predictions = {}
            pred_num = 0
            gold_sen_list = []
            pred_sen_list = []
            with tqdm(total=len(test_dataloader), file=sys.stdout) as pbar:
                for step, batch in enumerate(test_dataloader):

                    # add batch to gpu
                    batch = tuple(t.to(device) for t in batch)
                    b_full_input_ids, b_full_input_mask, _, b_prompt_input_ids, b_prompt_input_mask = batch
                    # print(b_input_ids.shape, b_input_mask.shape, b_labels.shape)

                    # forward pass
                    with torch.no_grad():
                        generation_output = model.generate(
                            input_ids=b_prompt_input_ids, 
                            attention_mask=b_prompt_input_mask, 
                            top_k=50,
                            top_p=0.9,
                            do_sample=False,
                            max_length=args.max_length, 
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                        gold_sens = tokenizer.batch_decode(b_full_input_ids, skip_special_tokens=True)
                        pred_sens = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

                        gold_sen_list += gold_sens
                        pred_sen_list += pred_sens
                        # print(len(gold_sens), gold_sens[0])
                        # print(len(pred_sens), pred_sens[0])

                    for gold_sen, pred_sen in zip(gold_sens, pred_sens):

                        # gold_sen = gold_sens[0]
                        gold_str = gold_sen.split("Answer: ")[1]
                        gold_antecedents = [
                            p.strip().lower() for p in gold_str.strip().split(SEP_TOKEN)
                        ]

                        # pred_sen = pred_sens[0]
                        predictions_str = pred_sen.split("Answer: ")[1]
                        predicted_antecedents = [
                            p.strip().lower() for p in predictions_str.strip().split(SEP_TOKEN)
                        ]
                        predicted_antecedents = list(filter(lambda a: a != "", predicted_antecedents))
                        predicted_antecedents = list(set(predicted_antecedents))

                        # print(gold_antecedents)
                        # print(predicted_antecedents)

                        predictions[pred_num] = {"gold_antecedents": gold_antecedents, "predictions": predicted_antecedents}
                        pred_num += 1

                        # break
                    pbar.update(1)

            # final compute metrics and save outputs
            metrics = compute_evaluation_metrics(predictions)
            metrics["num_predictions"] = len(predictions.keys())
            print(f"Test lenient F1: {metrics['lenient_f1']}")
            print(metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/axcell/selected.json", type=str, required=False)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output directory")
    parser.add_argument("--max_length", type=int, default=1024, help="max length of the generated text")
    parser.add_argument("--learning_rate", type=float, default=0.00002, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_grad_norm", type=int, default=1, help="max gradient norm")
    parser.add_argument("--n_shots", type=int, default=1, help="number of shots")

    args = parser.parse_args()
    print(args)

    main(args)
