#!/bin/bash 
export PYTHONPATH=. 

exp_dir=k32trial0_dev
test_data=../data/ChemuRef/lm_trimmed_data/dev.jsonl
prompt_combine_method=mixture_model

python evaluations/chemuref_evaluation.py $test_data $exp_dir $prompt_combine_method 