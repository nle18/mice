
trial_num=0
exp_dir=output/wo_title/k32trial"$trial_num"_dev
train_data=data/ChemuRef/lm_trimmed_data/train/k32/trial"$trial_num".jsonl
test_data=data/ChemuRef/lm_trimmed_data/dev.jsonl
prompt_num=256
stride_num=8

# python max/inference/gpt_j_local.py "$exp_dir" "$train_data" "$test_data" 0793-00 "$prompt_num"
echo python max/inference/gpt_j_local.py "$exp_dir" "$train_data" "$test_data" 0 "$prompt_num" $stride_num
python max/inference/gpt_j_local.py "$exp_dir" "$train_data" "$test_data" 0 "$prompt_num" $stride_num
