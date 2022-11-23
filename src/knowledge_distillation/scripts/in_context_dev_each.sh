
TRIAL_NUM=4
JOB_NAME="dev"
OUTPUT_PATH="k32trial${TRIAL_NUM}_dev/outpupts/0.out"
ERROR_PATH="k32trial${TRIAL_NUM}_dev/outpupts/0.err"
TRAIN_PATH="data/ChemuRef/lm_trimmed_data/train/k32/trial${TRIAL_NUM}.jsonl"
TEST_PATH="data/ChemuRef/lm_trimmed_data/dev.jsonl"
# TEST_ID="0793-00"
TEST_ID=0
EXP_DIR="k32trial${TRIAL_NUM}_dev/examples"
K=10
NUM_PROMPT=100
TACTIC=2
GPU_ID=0

echo python max/in_context_retrieval_methods/topk_bruteforce_predictions_truncatedContext.py $TRAIN_PATH $TEST_PATH $TEST_ID $EXP_DIR $K $NUM_PROMPT $TACTIC $GPU_ID
python max/in_context_retrieval_methods/topk_bruteforce_predictions_truncatedContext.py $TRAIN_PATH $TEST_PATH $TEST_ID $EXP_DIR $K $NUM_PROMPT $TACTIC $GPU_ID