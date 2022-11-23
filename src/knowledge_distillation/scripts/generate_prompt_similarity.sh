
# trial_num=0
# # exp_dir=output/wo_title/k32trial"$trial_num"_dev
# exp_dir=fan/output/unlabeled_best_trial

# # train_data=data/ChemuRef/lm_trimmed_data/train/k32/trial"$trial_num".jsonl
# train_data=data/ChemuRef_v2/lm_trimmed_data/train/k32/trial"$trial_num".jsonl
# # test_data=data/ChemuRef/lm_trimmed_data/dev.jsonl
# test_data=data/unlabeled/uspto_app_sample_2000_chemref_anaphora_pred.jsonl
# type=similar
# # type=random
# num_prompts=100
# encoder=roberta

trial_num=3
exp_dir=fan/output/unlabeled_median_trial/
train_data=data/ChemuRef_v3/lm_trimmed_data/train/k32/trial"$trial_num".jsonl
test_data=data/unlabeled/uspto_5000_paras_anaphor_pred_chemref_0_2000.jsonl
type=random
num_prompts=256
encoder=roberta

python max/prompt_generation/generate_prompts.py $exp_dir $train_data $test_data $type $num_prompts $encoder