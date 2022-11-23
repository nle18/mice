MODEL='t5-3b'
# MODEL='bigscience/T0_3B'

# N_SHOT=64
# N_SHOT=32
# N_SHOT=16
# N_SHOT=8
N_SHOT=4

python code/t0_finetune.py \
    --data_dir /srv/share5/nghia6/data/ChemuRef/lm_trimmed_data \
    --model_name_or_path "$MODEL" \
    --output_dir outputs \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --max_length 1024 \
    --n_shots "$N_SHOT" \
    --n_epochs 20