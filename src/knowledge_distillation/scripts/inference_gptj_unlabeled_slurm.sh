
#!/bin/bash

# trial_num=0
# exp_dir=fan/output/unlabeled_best_trial_w_title/
# train_data=data/ChemuRef_v2/lm_trimmed_data/train/k32/trial"$trial_num".jsonl
# test_data=data/unlabeled/uspto_app_sample_2000_chemref_anaphora_pred.jsonl
# prompt_num=100
# stride_num=13
# gpu_id=0

# start_idx=400
# end_idx=499

trial_num=3
exp_dir=fan/output/unlabeled_median_trial/
train_data=data/ChemuRef_v3/lm_trimmed_data/train/k32/trial"$trial_num".jsonl
test_data=data/unlabeled/uspto_5000_paras_anaphor_pred_chemref_0_2000.jsonl
prompt_num=256
stride_num=100
gpu_id=0

start_idx=0
end_idx=799

while [ $start_idx -le $end_idx ]
do

    screenName="infer_trial_${trial_num}_start_idx_${start_idx}"

    echo $screenName
    screen -dmS "$screenName" sh
    screen -S "$screenName" -X stuff "srun --constraint a40 --gres gpu:1 -p short -c 6  --pty bash 
    "
    screen -S "$screenName" -X stuff "bash
    "
    screen -S "$screenName" -X stuff "conda activate gptj-gpu
    "
    screen -S "$screenName" -X stuff "echo python fan/code/gpt_j_protocol.py $exp_dir $train_data $test_data $start_idx $prompt_num $stride_num $gpu_id
    "
    screen -S "$screenName" -X stuff "python fan/code/gpt_j_protocol.py $exp_dir $train_data $test_data $start_idx $prompt_num $stride_num $gpu_id
    "

    sleep 3s
    # start_idx=$(($start_idx+13))
    start_idx=$((start_idx+stride_num))

done
