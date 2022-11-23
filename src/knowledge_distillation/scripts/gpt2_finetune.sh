# python code/gpt2_finetune.py \
#     --data_dir /srv/share5/nghia6/data/ChemuRef/lm_trimmed_data \
#     --model_name_or_path gpt2-xl \
#     --output_dir outputs \
#     --learning_rate 2e-5 \
#     --batch_size 1 \
#     --max_length 1024 \
#     --n_shots 64 \
#     --n_epochs 100

python code/gpt2_finetune.py \
    --data_dir /srv/share5/nghia6/data/ChemuRef/lm_trimmed_data \
    --model_name_or_path gpt2-xl \
    --output_dir outputs \
    --learning_rate 2e-5 \
    --batch_size 1 \
    --max_length 1024 \
    --n_shots 4 \
    --n_epochs 50