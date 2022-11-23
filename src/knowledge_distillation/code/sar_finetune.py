

from dataclasses import replace
import math
import glob
import time
import json
import pickle
import os
import numpy as np
import operator
import time
import sys
import random
import argparse

from transformers import AdamW
from tqdm import tqdm, trange

import torch

from collections import defaultdict, Counter, OrderedDict
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertConfig, BertTokenizer, BertPreTrainedModel, AdamW
from transformers import RobertaTokenizer, RobertaForTokenClassification

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from conlleval import evaluate
from chemuref_evaluation import *

class BertForSAR(BertForTokenClassification):

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            labels=None, head_tags=None, head_flags=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        #         print("sequence_output.size()", sequence_output.size())
        #         print("labels.size()", labels.size())
        #         print("attention_mask.size()", attention_mask.size())
        #         print("head_flags.size()", head_flags.size())
        #         print("head_flags", head_flags)
        #         print("attention_mask", attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        #         print("logits.size()", logits.size())

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:

                if head_tags is None:
                    active_loss = attention_mask.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                #                 print("active_labels.size()", active_labels.size())
                else:
                    active_loss = head_flags.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, head_tags.view(-1), torch.tensor(loss_fct.ignore_index).type_as(head_tags)
                    )

                active_logits = logits.view(-1, self.num_labels)
                #                 print("active_logits.size()", active_logits.size())

                loss = loss_fct(active_logits, active_labels)
            #                 print("loss.size()", loss.size())
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

CHEMREF_ENT_NAME = ['Antecedent']
USE_HEAD_ONLY = True

def generate_tag2id(ent_name_list):

    tag_name_list = ["O"]

    for each_ent_name in ent_name_list:
        tag_name_list.append('B-' + each_ent_name)
        tag_name_list.append('I-' + each_ent_name)

    tag2id = dict([(value, key) for key, value in enumerate(tag_name_list)])

    return tag_name_list, tag2id

CHEMREF_TAG_NAME, CHEMREF_TAG2IDX = generate_tag2id(CHEMREF_ENT_NAME)

ENT_NAME = {
    'chemref': CHEMREF_ENT_NAME,
}

TAG_NAME = {
    'chemref': CHEMREF_TAG_NAME,
}

TAG2IDX = {
    'chemref': CHEMREF_TAG2IDX,
}

CHEMREF_PATH = {
    'train': 'data/ChemuRef_v3/lm_trimmed_data/train/k32/trial3.jsonl',
    # 'train_pseudo': 'fan/output/unlabeled_best_trial/countBased_predictions.json',
    'dev': 'data/ChemuRef_v3/lm_trimmed_data/dev.jsonl',
}

DATA_PATH = {
    'chemref': CHEMREF_PATH,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--lm_model", default=None, type=str, required=True)
    parser.add_argument('--ckp_num', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_sen', type=int, default=10)
    parser.add_argument('--n_labels', type=int)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    parser.add_argument('--max_len', type=int, default=256, required=True)
    parser.add_argument('--patient', type=int, default=30)
    parser.add_argument('--eval_step', type=int, default=0)
    parser.add_argument('--budget', type=int, default=0)
    parser.add_argument('--num_subset_file', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--down_sample_rate', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument("--gpu_ids", default=None, type=str, required=True)
    parser.add_argument("--task_name", default='', type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=1234,
                        help="random seed for random library")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--save_model", action='store_true', help="Save trained checkpoints.")
    parser.add_argument("--down_sample", action='store_true', help="Sample the negative data in the training.")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--src_data', type=str, default='')
    parser.add_argument('--tgt_data', type=str, default='')
    parser.add_argument('--data_name', type=str, default='')
    parser.add_argument('--train_source', type=str, default='gold')
    # parser.add_argument('--train_path', type=str, default='')
    # parser.add_argument('--dev_path', type=str, default='')
    # parser.add_argument('--test_path', type=str, default='')

    args = parser.parse_args()

    return args


def get_model(args):

    if args.lm_model == "bert":
        model_name = "bert-base-uncased"
    elif args.lm_model == "bertlarge":
        model_name = "bert-large-uncased"
    elif args.lm_model == "roberta":
        model_name = "roberta-base"
    elif args.lm_model == "robertalarge":
        model_name = "roberta-large"
    elif args.lm_model == "biomed":
        model_name = "allenai/biomed_roberta_base"
    elif args.lm_model == 'scibert':
        model_name = "allenai/scibert_scivocab_uncased"
    elif args.lm_model == "procroberta":
        model_name = "fbaigt/proc_roberta"
    elif args.lm_model == "procbert":
        model_name = "fbaigt/procbert"
    else:
        raise

    print("Load the model checkpoint from: ", model_name)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = args.n_labels
    if args.task_name == 'sar':
        model = BertForSAR.from_pretrained(model_name, config=config)
    else:
        raise

    
    if args.train_source == 'gold':
        exp_setup = f'gold_lr_{args.learning_rate}_epoch_{args.epochs}_bs_{args.batch_size}_maxlen_{args.max_len}'
    elif args.train_source == 'pseudo':
        exp_setup = f'pseudo_lr_{args.learning_rate}_epoch_{args.epochs}_bs_{args.batch_size}_maxlen_{args.max_len}'
    else:
        raise
        
    saved_model_dir = f"{args.output_dir}/{args.task_name}/{args.lm_model}/{exp_setup}/seed_{args.random_seed}"

    return tokenizer, model, model_name, saved_model_dir



def get_processed_sentences(data_name, data_class, max_len, batch_size, tokenizer):

    data_path = DATA_PATH[data_name][data_class]
    # file_order = FILE_ORDER[data_name][data_class]
    ent_name = ENT_NAME[data_name]
    tag2idx = TAG2IDX[data_name]

    tokenized_word_list = []
    tokenized_label_list = []
    head_label_list = []
    head_flag_list = []

    untokenized_word_list = []
    example_id_list = []
    long_flag_list = []

    example_list = []

    if data_class == 'train_pseudo':

        # load anaphor data
        anaphor_dict = {}
        with open("data/unlabeled/uspto_app_sample_2000_chemref_anaphora_pred.jsonl", "r") as f:
            for line in f.readlines():
                example = json.loads(line)
                anaphor_dict[example["example_id"]] = example["anaphor"]

        with open(data_path, "r") as f:
            pseudo_data = json.load(f)
            for example_id, example_data in pseudo_data.items():
                example = {}
                example["example_id"] = example_id
                example["context"] = example_data["context"]
                assert example_id in anaphor_dict
                example["anaphor"] = anaphor_dict[example_id]
                example["gold_antecedents"] = example_data["predicted_antecedents"]
                example_list.append(example)

    else:
        print(f"Load data from: {data_path}")
        with open(data_path, "r") as f:
            for line in f.readlines():
                example = json.loads(line)
                example_list.append(example)

    for example in tqdm(example_list):
        example_id = example["example_id"]
        context = example["context"]
        anaphor = example["anaphor"]
        antecedents = example["gold_antecedents"]

        context = context.lower()
        anaphor = anaphor.lower()
        antecedents = [antecedent.lower() for antecedent in antecedents]

        # print(f"example_id: {example_id}")
        # print(f"context: {context}")
        # print(f"anaphor: {anaphor}")
        # print(f"antecedents: {antecedents}")

        # TODO: Figure out the duplicate antecedents (get char index of each antecedent)
        if len(antecedents) != len(set(antecedents)):
            print("Duplicate antecedents!")
            # print(antecedents)
            # break
            # continue
        
        assert context.endswith(".")
        context = context[:-1]

        assert context.endswith(anaphor)
        anaphor_char_start = len(context) - len(anaphor)
        anaphor_char_end = len(context)
        assert context[anaphor_char_start:anaphor_char_end] == anaphor

        if sum([antecedent in context for antecedent in antecedents]) != len(antecedents):
            print("Some antecedents are not in the context!")
            print("context:", context)
            print("antecedents:", antecedents)
            
        assert sum([antecedent in context for antecedent in antecedents]) == len(antecedents)
        antecedent_char_idx_list = [[context.index(antecedent), context.index(antecedent)+len(antecedent), antecedent] for antecedent in antecedents]
        # print(antecedent_char_idx_list)
        antecedent_char_idx_list = sorted(antecedent_char_idx_list, key=lambda x: x[0])
        # print(antecedent_char_idx_list)
        for antecedent_char_start, antecedent_char_end, antecedent in antecedent_char_idx_list:
            # print(antecedent_char_start, antecedent_char_end)
            # print(f"antecedent: {context[antecedent_char_start:antecedent_char_end]}")
            # print(antecedent)
            assert context[antecedent_char_start:antecedent_char_end] == antecedent

        context_nltk_tokenized = word_tokenize(context)
        # print(f"context_nltk_tokenized: {context_nltk_tokenized}")
        assert ''.join(context_nltk_tokenized) == context.replace(" ", "")

        char2word_index = {}
        char_idx_org = 0
        for word_idx, word in enumerate(context_nltk_tokenized):
            for char_idx, char in enumerate(word):
                # print(f"char: {char}")
                # print(f"char_idx_org: {char_idx_org}")
                # print(f"context[char_idx_org]: {context[char_idx_org]}")
                while char_idx_org < len(context) and context[char_idx_org] == ' ':
                    char_idx_org += 1
                if char == context[char_idx_org]:
                    char2word_index[char_idx_org] = word_idx
                    char_idx_org += 1
                else:
                    raise
        
        anaphor_word_start = char2word_index[anaphor_char_start]
        anaphor_word_end = char2word_index[anaphor_char_end-1]
        anaphor_nltk_tokenized = context_nltk_tokenized[anaphor_word_start:anaphor_word_end+1]
        assert ''.join(anaphor_nltk_tokenized) == anaphor.replace(" ", "")
        # print(f"tokenized anaphor: {anaphor_nltk_tokenized}")

        # Add marker in context to specify the position of anaphor
        context_nltk_tokenized_anaphor = context_nltk_tokenized[:anaphor_word_start] + ["[ana-start]"] + context_nltk_tokenized[anaphor_word_start:anaphor_word_end+1] + ["[ana-end]"]
        # context_nltk_tokenized_anaphor = context_nltk_tokenized
        # print(f"context_nltk_tokenized_anaphor: {context_nltk_tokenized_anaphor}")

        label_list = ["O"] * len(context_nltk_tokenized_anaphor)
        for antecedent_char_start, antecedent_char_end, antecedent in antecedent_char_idx_list:
            antecedent_word_start = char2word_index[antecedent_char_start]
            antecedent_word_end = char2word_index[antecedent_char_end-1]
            antecedent_nltk_tokenized = context_nltk_tokenized[antecedent_word_start:antecedent_word_end+1]
            if ''.join(antecedent_nltk_tokenized) != antecedent.replace(" ", ""):
                print(f"antecedent_nltk_tokenized: {antecedent_nltk_tokenized}")
                print(f"original antecedent: {antecedent}")
            # assert ''.join(antecedent_nltk_tokenized) == antecedent.replace(" ", "")
            # print(f"tokenized antecedent: {antecedent_nltk_tokenized}")

            label_list[antecedent_word_start] = "B-Antecedent"
            for word_idx in range(antecedent_word_start+1, antecedent_word_end+1):
                label_list[word_idx] = "I-Antecedent"

        word_list = context_nltk_tokenized_anaphor
        assert len(word_list) == len(label_list)
        # print(f"word/label list: {list(zip(word_list, label_list))}")

        def get_subtoken_label(word_list, label_list):

            piece_list_all = []
            flag_list_all = []
            head_label_list_all = []
            piece_label_list_all = []

            for word, word_label in zip(word_list, label_list):

                piece_list = tokenizer.tokenize(word)
                piece_label_list = [word_label] + [word_label.replace("B-", "I-")] * (
                            len(piece_list) - 1) if word_label.startswith("B-") else \
                    [word_label] * len(piece_list)
                flag_list = [1] + [0] * (len(piece_list) - 1)
                head_label_list = [word_label] + ["O"] * (len(piece_list) - 1)

                piece_list_all += piece_list
                piece_label_list_all += piece_label_list

                flag_list_all += flag_list
                head_label_list_all += head_label_list

            assert len(word_list) == sum(flag_list_all)
            assert len(flag_list_all) == len(head_label_list_all)
            assert len(piece_list_all) == len(flag_list_all)
            assert len(piece_list_all) == len(piece_label_list_all)

            # Add "cls" and "eos" for RobertaTokenizer
            piece_list_all = [tokenizer.cls_token] + piece_list_all + [tokenizer.eos_token]

            piece_label_list_all = ["O"] + piece_label_list_all + ["O"]
            head_label_list_all = ["O"] + head_label_list_all + ["O"]
            flag_list_all = [0] + flag_list_all + [0]

            assert len(flag_list_all) == len(head_label_list_all)
            assert len(piece_list_all) == len(flag_list_all)
            assert len(piece_list_all) == len(piece_label_list_all)

            return piece_list_all, piece_label_list_all, head_label_list_all, flag_list_all

        tokenized_word, tokenized_label, head_label, head_flag = get_subtoken_label(word_list, label_list)

        tokenized_word_list.append(tokenized_word)
        tokenized_label_list.append(tokenized_label)
        head_label_list.append(head_label)
        head_flag_list.append(head_flag)

        untokenized_word_list.append(word_list)
        example_id_list.append(example_id)
        long_flag_list.append(len(tokenized_word) > max_len)
        # print()

    print(f"The number of sentences: {len(tokenized_word_list)}, the max sentence length: {max([len(item) for item in tokenized_word_list])}")

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_word_list], \
                              maxlen=max_len, value=tokenizer.pad_token_id, dtype="long", \
                              truncating="post", padding="post")
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    tags = pad_sequences([[tag2idx[l] for l in lab] for lab in tokenized_label_list],
                         maxlen=max_len, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")
    tags = torch.tensor(tags)

    head_tags = pad_sequences([[tag2idx[l] for l in lab] for lab in head_label_list],
                              maxlen=max_len, value=tag2idx["O"], padding="post",
                              dtype="long", truncating="post")
    head_tags = torch.tensor(head_tags)

    head_flags = pad_sequences(head_flag_list,
                               maxlen=max_len, value=0, padding="post",
                               dtype="long", truncating="post")
    head_flags = torch.tensor(head_flags)

    final_data = TensorDataset(inputs, masks, tags, head_tags, head_flags)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader, masks, tags, head_tags, head_flags, untokenized_word_list, example_id_list, long_flag_list


def index_ent_in_prediction(word_list, tag_list):
    ent_queue, ent_idx_queue, ent_type_queue = [], [], []
    ent_list, ent_idx_list, ent_type_list = [], [], []

    for word_idx in range(len(word_list)):

        if 'B-' in tag_list[word_idx]:
            if ent_queue:

                if len(set(ent_type_queue)) != 1:
                    print(ent_queue)
                    print(ent_idx_queue)
                    print(ent_type_queue)
                    print(Counter(ent_type_queue).most_common())
                    print()

                else:
                    ent_list.append(' '.join(ent_queue).strip())
                    #                     ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                    ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                    assert len(set(ent_type_queue)) == 1
                    ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []
            ent_queue.append(word_list[word_idx])
            ent_idx_queue.append(word_idx)
            ent_type_queue.append(tag_list[word_idx][2:])

        if 'I-' in tag_list[word_idx]:
            if word_idx == 0 or (word_idx > 0 and tag_list[word_idx][2:] == tag_list[word_idx - 1][2:]):
                ent_queue.append(word_list[word_idx])
                ent_idx_queue.append(word_idx)
                ent_type_queue.append(tag_list[word_idx][2:])
            else:
                if ent_queue:

                    if len(set(ent_type_queue)) != 1:
                        print(ent_queue)
                        print(ent_idx_queue)
                        print(ent_type_queue)
                        print(Counter(ent_type_queue).most_common())
                        print()
                    else:
                        ent_list.append(' '.join(ent_queue).strip())
                        #                         ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                        ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                        assert len(set(ent_type_queue)) == 1
                        ent_type_list.append(ent_type_queue[0])

                ent_queue, ent_idx_queue, ent_type_queue = [], [], []
                ent_queue.append(word_list[word_idx])
                ent_idx_queue.append(word_idx)
                ent_type_queue.append(tag_list[word_idx][2:])

        if 'O' == tag_list[word_idx] or word_idx == len(word_list) - 1:
            if ent_queue:

                if len(set(ent_type_queue)) != 1:
                    print(ent_queue)
                    print(ent_idx_queue)
                    print(ent_type_queue)
                    print(Counter(ent_type_queue).most_common())
                    print()

                else:
                    ent_list.append(' '.join(ent_queue).strip())
                    #                     ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                    ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                    assert len(set(ent_type_queue)) == 1
                    ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []

    return ent_list, ent_idx_list, ent_type_list


def main(args):
    """
    Main function
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.random_seed)

    args.n_labels = len(TAG2IDX[args.data_name])
    tag_name = TAG_NAME[args.data_name]
    print(f"The number of labels: {args.n_labels}")
    print(f"The tag name: {tag_name}")
    tokenizer, model, model_name, saved_model_dir = get_model(args)

    tokenizer.add_tokens(["[ana-start]", "[ana-end]"])
    # Resize the model
    model.resize_token_embeddings(len(tokenizer))

    # Make the path if it doesn't exist
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    result_path = os.path.join(saved_model_dir, 'results.json')

    if args.train_source == "gold":
        train_dataloader, train_masks, train_tags, \
        train_head_tags, train_head_flags, train_untokenized_words, \
        train_file_ids, train_long_flags = get_processed_sentences(args.data_name, 'train', args.max_len, args.batch_size, tokenizer)
    elif args.train_source == "pseudo":
        train_dataloader, train_masks, train_tags, \
        train_head_tags, train_head_flags, train_untokenized_words, \
        train_file_ids, train_long_flags = get_processed_sentences(args.data_name, 'train_pseudo', args.max_len, args.batch_size, tokenizer)

    dev_dataloader, dev_masks, dev_tags, \
    dev_head_tags, dev_head_flags, dev_untokenized_words, \
    dev_file_ids, dev_long_flags = get_processed_sentences(args.data_name, 'dev', args.max_len, args.batch_size, tokenizer)

    # model.cuda();
    if n_gpu > 1:
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = 0
    config = vars(args).copy()
    config['saved_model_dir'] = saved_model_dir

    for num_epoch in trange(args.epochs, desc="Epoch"):

        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):

            if step % 100 == 0 and step > 0:
                print("The number of steps: {}".format(step))

            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_head_tags, b_head_flags = batch
            # forward pass
            if not USE_HEAD_ONLY:
                loss, _ = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            else:
                loss, _ = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels,
                                head_tags=b_head_tags, head_flags=b_head_flags)

            if n_gpu > 1:
                loss = loss.mean()

            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # on dev set
        model.eval()
        predictions, true_labels = [], []
        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_head_tags, b_head_flags = batch

            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

        if not USE_HEAD_ONLY:
            pred_tags_str = [tag_name[p_i] for p_idx, p in enumerate(predictions)
                             for p_i_idx, p_i in enumerate(p)
                             if dev_masks[p_idx][p_i_idx]
                             ]
            dev_tags_str = [tag_name[l_i.tolist()] for l_idx, l in enumerate(dev_tags)
                            for l_i_idx, l_i in enumerate(l)
                            if dev_masks[l_idx][l_i_idx]
                            ]
        else:
            pred_tags_str = [tag_name[p_i] for p_idx, p in enumerate(predictions)
                             for p_i_idx, p_i in enumerate(p)
                             if dev_head_flags[p_idx][p_i_idx]
                             ]
            dev_tags_str = [tag_name[l_i.tolist()] for l_idx, l in enumerate(dev_head_tags)
                            for l_i_idx, l_i in enumerate(l)
                            if dev_head_flags[l_idx][l_i_idx]
                            ]

        if USE_HEAD_ONLY:
            pred_tags_str = [[tag_name[p_i] for p_i_idx, p_i in enumerate(p) if dev_head_flags[p_idx][p_i_idx]] for p_idx, p in enumerate(predictions)]
            dev_tags_str = [[tag_name[l_i.tolist()] for l_i_idx, l_i in enumerate(l) if dev_head_flags[l_idx][l_i_idx]] for l_idx, l in enumerate(dev_head_tags)]


        # # https://github.com/sighsmile/conlleval
        # prec_dev, rec_dev, f1_dev = evaluate(dev_tags_str, pred_tags_str, verbose=False)
        # print("\nOn dev set: ")
        # print("Precision-Score: {}".format(prec_dev))
        # print("Recall-Score: {}".format(rec_dev))
        # print("F1-Score: {}".format(f1_dev))
        # print()

        # Max's evaluation
        def read_jsonl(filepath: str) -> dict:
            data = dict()
            with open(filepath, "r") as f:
                for line in f.readlines():
                    example = json.loads(line)
                    data[example["example_id"]] = example
            return data

        gold_data = read_jsonl(DATA_PATH[args.data_name]['dev'])
       
        predictions = {}
        for example_id, words, pred_tag, gold_tag, long_flag in zip(dev_file_ids, dev_untokenized_words, pred_tags_str, dev_tags_str, dev_long_flags):
            assert len(words) == len(pred_tag)
            words = [each_word for each_word in words if not each_word.startswith("[ana-")]

            context = gold_data[example_id]["context"].rstrip(".")
            context = context.lower()
            context_remove_space = context.replace(" ", "")
            # print(context_remove_space)
            # print("".join(words))
            assert "".join(words) == context_remove_space

            char_idx_dict = {}
            char_idx_keep_space, char_idx_remove_space = 0, 0
            for char in context:
                if char != " ":
                    # print(char, context_remove_space[char_idx_remove_space])
                    assert char == context_remove_space[char_idx_remove_space]
                    char_idx_dict[char_idx_remove_space] = char_idx_keep_space
                    char_idx_keep_space += 1
                    char_idx_remove_space += 1
                else:
                    char_idx_keep_space += 1

            ent_list, ent_idx_list, ent_type_list = index_ent_in_prediction(words, pred_tag)
            predicted_antecedents = []
            # print(ent_list, ent_idx_list, ent_type_list)
            for each_ent in ent_list:
                # print(each_ent)
                each_ent = each_ent.replace(" ", "")
                ent_start_idx = context_remove_space.find(each_ent)
                ent_end_idx = ent_start_idx + len(each_ent)
                assert each_ent == context_remove_space[ent_start_idx:ent_end_idx]
                ent_start_char_idx = char_idx_dict[ent_start_idx]
                ent_end_char_idx = char_idx_dict[ent_end_idx-1]+1
                each_ent_org = context[ent_start_char_idx:ent_end_char_idx]
                assert each_ent_org.replace(" ", "") == each_ent

                if len(each_ent_org) > 1:
                    predicted_antecedents.append(each_ent_org)
                # print(each_ent_org)
                # print()

            gold_antecedents = gold_data[example_id]["gold_antecedents"]
            gold_antecedents = [gold_antecedent.lower() for gold_antecedent in gold_antecedents]
            predictions[example_id] = {"predicted_antecedents": predicted_antecedents, "gold_antecedents": gold_antecedents}
            # print("gold_antecedents", gold_antecedents)
            # print("predicted_antecedents", predicted_antecedents)
            # print()

        # predictions = gold_data
        # for example_id, prediction in predictions.items():
        #     prediction["predicted_antecedents"] = prediction["gold_antecedents"]

        # compute evaluation metrics
        span_single_metrics = compute_evaluation_metrics_span_single(predictions, gold_data)
        # print(span_single_metrics)

        f1_dev = span_single_metrics["span_single_f1"]

        if f1_dev > best_f1:
            best_f1 = f1_dev

            config['precision_dev'] = span_single_metrics["span_single_precision"]
            config['recall_dev'] = span_single_metrics["span_single_recall"]
            config['f1_dev'] = f1_dev

            config['best_epoch'] = num_epoch

            print(config, '\n')

            # Save hyper-parameters (lr, batch_size, epoch, precision, recall, f1)
            with open(result_path, 'w') as json_file:
                json.dump(config, json_file)

if __name__ == '__main__':

    args = get_args()
    main(args)

    # sh sar_finetune.sh 
