
#!/bin/bash

trial_num=0
exp_dir=output/wo_title/k32trial"$trial_num"_dev
train_data=data/ChemuRef/lm_trimmed_data/train/k32/trial"$trial_num".jsonl
test_data=data/ChemuRef/lm_trimmed_data/dev.jsonl
prompt_num=256
stride_num=8

start_idx=0
end_idx=0

while [ $start_idx -le $end_idx ]
do

    screenName="infer_trial_${trial_num}_start_idx_${start_idx}"

    echo $screenName
    screen -dmS "$screenName" sh
    screen -S "$screenName" -X stuff "bash
    "
    screen -S "$screenName" -X stuff "conda activate gptj-gpu
    "
    screen -S "$screenName" -X stuff "echo python max/inference/gpt_j_local.py $exp_dir $train_data $test_data $start_idx $prompt_num $stride_num
    "
    screen -S "$screenName" -X stuff "python max/inference/gpt_j_local.py $exp_dir $train_data $test_data $start_idx $prompt_num $stride_num
    "

    sleep 3s
    start_idx=$(($start_idx+8))

done
