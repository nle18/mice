exp_dir=fan/output/unlabeled_best_trial
test_data=data/unlabeled/uspto_app_sample_2000_chemref_anaphora_pred.jsonl
p_m_pre=0.1
p_m_post=0.1

python max/prompt_combination/count_based.py $exp_dir $test_data $p_m_pre $p_m_post