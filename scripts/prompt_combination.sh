#!/bin/bash 
export PYTHONPATH=.

exp_dir=k32trial0_dev
test_data=../data/ChemuRef/lm_trimmed_data/dev.jsonl
p_m_pre=0.01
p_m_post=0.02
prior_distribution=similar 
similarScores_combine=addition

python prompt_combination/mixture_model.py $exp_dir $test_data $p_m_pre $p_m_post $prior_distribution $similarScores_combine 
