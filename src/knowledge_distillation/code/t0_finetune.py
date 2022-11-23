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

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import  T5Tokenizer, T5ForConditionalGeneration, AdamW

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

def generate_data_answer(data_item, tokenizer):
    # return f"{SEP_TOKEN.join(data_item['gold_antecedents'])}{EOS_TOKEN}"
    # return SEP_TOKEN.join(data_item['gold_antecedents'])
    return "|".join(data_item['gold_antecedents'])+tokenizer.eos_token

def load_data(data_path, input_tokenizer, args, data_type="train"):

    data_list = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    data_full_list = []
    data_prompt_list = []
    data_answer_list = []
    # print(data_list[0])

    for data_item in data_list:
        # data_full = generate_data_full(data_item)
        data_prompt = generate_data_prompt(data_item)
        data_answer = generate_data_answer(data_item, input_tokenizer)

        # data_full_list.append(data_full)
        data_prompt_list.append(data_prompt)
        data_answer_list.append(data_answer)

    # data_full = input_tokenizer(data_full_list, padding=True, max_length=args.max_length, truncation=True, return_tensors='pt')
    source_encoding = input_tokenizer(data_prompt_list, padding=True, max_length=args.max_length, truncation=True, return_tensors='pt')
    input_ids, attention_mask = source_encoding.input_ids, source_encoding.attention_mask

    target_encoding = input_tokenizer(data_answer_list, padding=True, max_length=256, truncation=True, return_tensors='pt')
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100
    # In PyTorch and Tensorflow, -100 is the ignore_index of the
    labels = torch.tensor(labels)
    labels[labels == input_tokenizer.pad_token_id] = -100

    print(input_ids.size(), attention_mask.size(), labels.size())
    assert(labels.size()[1] < args.max_length), f"Increaset the max target length to : {1+labels.size()[1]}"

    dataset = TensorDataset(input_ids, attention_mask, labels)
    data_sampler = RandomSampler(dataset) if data_type == "train" else SequentialSampler(dataset)
    data_dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size) if data_type == "train" else DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size*4)

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


def get_model(args):
    print("Load checkpoints form: ", args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer.add_tokens(['<outside>', '<create>', '<exist>', '<move>', '<destroy>']) # added to maintain reproducibility of previous experiments
    # tokenizer.add_tokens([f'<start>', f'<options>', f'</options>']) # add start state
    # tokenizer.add_special_tokens({'sep_token': SEP_TOKEN})
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def main(args):
    
    # Set seed
    set_seed(args)
    
    # Load tokenizer and model
    tokenizer, model = get_model(args)

    device_map = {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10, 11], 2: [12, 13, 14, 15, 16, 17],
                  3: [18, 19, 20, 21, 22, 23]}
    model.parallelize(device_map)

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

    # # Load model
    # print("Loading model...\n")
    # model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, pad_token_id=tokenizer.pad_token_id)
    # model.resize_token_embeddings(len(tokenizer))

    # Config the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # if n_gpu > 1:
    #     model.to(device)
    #     model = torch.nn.DataParallel(model)
    # else:
    #     model.cuda()

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
                b_input_ids, b_input_mask, b_labels = batch

                # forward pass
                outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss

                # if n_gpu > 1:
                #     loss = loss.mean()

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
        # dev_gold = []
        # dev_pred = []
        with tqdm(total=len(dev_dataloader), file=sys.stdout) as pbar:
            for step, batch in enumerate(dev_dataloader):

                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # print(b_input_ids.shape, b_input_mask.shape, b_labels.shape)

                # forward pass
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=b_input_ids, 
                        attention_mask=b_input_mask,
                        max_length=128
                    )
                
                    gold_sens = tokenizer.batch_decode([[label if label != -100 else 1 for label in labels] for labels in b_labels], skip_special_tokens=True)
                    pred_sens = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

                    # dev_gold += gold_sens
                    # dev_pred += pred_sens

                    # gold_sen_list += gold_sens
                    # pred_sen_list += pred_sens
                    
                    # print(len(gold_sens), gold_sens[0])
                    # print(len(pred_sens), pred_sens[0])

                for gold_sen, pred_sen in zip(gold_sens, pred_sens):
                    # print(f"gold_sen: {gold_sen}")
                    # print(f"pred_sen: {pred_sen}")
                    # gold_sen = gold_sens[0]
                    # gold_str = gold_sen.split("Answer: ")[1]
                    # gold_antecedents = [
                    #     p.strip().lower() for p in gold_str.strip().split(SEP_TOKEN)
                    # ]
                    # gold_antecedents = gold_sen.strip().split(SEP_TOKEN)
                    gold_antecedents = gold_sen.strip().split("|")

                    # pred_sen = pred_sens[0]
                    # predictions_str = pred_sen.split("Answer: ")[1]
                    # predicted_antecedents = [
                    #     p.strip().lower() for p in predictions_str.strip().split(SEP_TOKEN)
                    # ]
                    # predicted_antecedents = list(filter(lambda a: a != "", predicted_antecedents))
                    # predicted_antecedents = list(set(predicted_antecedents))
                    # predicted_antecedents = pred_sen.strip().split(SEP_TOKEN)
                    predicted_antecedents = pred_sen.strip().split("|")

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
            with tqdm(total=len(test_dataloader), file=sys.stdout) as pbar:
                for step, batch in enumerate(test_dataloader):

                    # add batch to gpu
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch
                    # print(b_input_ids.shape, b_input_mask.shape, b_labels.shape)

                    # forward pass
                    with torch.no_grad():
                        generation_output = model.generate(
                            input_ids=b_input_ids, 
                            attention_mask=b_input_mask,
                            max_length=128
                        )
                    
                        gold_sens = tokenizer.batch_decode([[label if label != -100 else 1 for label in labels] for labels in b_labels], skip_special_tokens=True)
                        pred_sens = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

                        # gold_sen_list += gold_sens
                        # pred_sen_list += pred_sens
                        # print(len(gold_sens), gold_sens[0])
                        # print(len(pred_sens), pred_sens[0])

                    for gold_sen, pred_sen in zip(gold_sens, pred_sens):

                        # # gold_sen = gold_sens[0]
                        # gold_str = gold_sen.split("Answer: ")[1]
                        # gold_antecedents = [
                        #     p.strip().lower() for p in gold_str.strip().split(SEP_TOKEN)
                        # ]
                        # gold_antecedents = gold_sen.strip().split(SEP_TOKEN)
                        gold_antecedents = gold_sen.strip().split("|")

                        # # pred_sen = pred_sens[0]
                        # predictions_str = pred_sen.split("Answer: ")[1]
                        # predicted_antecedents = [
                        #     p.strip().lower() for p in predictions_str.strip().split(SEP_TOKEN)
                        # ]
                        # predicted_antecedents = list(filter(lambda a: a != "", predicted_antecedents))
                        # predicted_antecedents = list(set(predicted_antecedents))
                        # predicted_antecedents = pred_sen.strip().split(SEP_TOKEN)
                        predicted_antecedents = pred_sen.strip().split("|")

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
