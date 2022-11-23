#!/bin/bash 

# Set PATH
export PYTHONPATH=.

exp_dir=k32trial0_dev # name of experiment folder
train_data=../data/ChemuRef/lm_trimmed_data/train/k32/trial0.jsonl
test_data=../data/ChemuRef/lm_trimmed_data/dev.jsonl
num_prompts=256
prompt_ordering=random
maxInContextNum=5

python prompt_generation/generate_variableNumInContext_prompts.py \
$exp_dir $train_data $test_data $num_prompts $type $maxInContextNum