
python fan/code/sar_finetune.py \
    --train_source gold \
    --lm_model procbert \
    --batch_size 32 \
    --max_len 512 \
    --gpu_ids 0 \
    --task_name sar \
    --data_name chemref \
    --epochs 200 \
    --output_dir fan/results
    