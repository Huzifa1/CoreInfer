MODEL_NAME='llama3-3b'
CHECKPOINT_PATH='./models/llama3-3b/'
SPARSITY='0.4'
TOKEN_SPARSITY='0.2'
NUM_PROMPTS='1000'

run_inference()
{
    echo RUN $METHOD for $TASK_NAME
    python evaluation/evaluate_task.py --model_name $MODEL_NAME --checkpoint_path $CHECKPOINT_PATH --sparsity $SPARSITY --token_sparsity $TOKEN_SPARSITY --limit $NUM_PROMPTS --method $METHOD --task_name $TASK_NAME --cpu_only --output_path ${MODEL_NAME}_${TASK_NAME}_${METHOD}.json
}

run_task()
{
    METHOD=dense
    run_inference
    
    METHOD=stable_guided
    run_inference
}

cd ./../

TASK_NAME=truthfulqa_gen
run_task

TASK_NAME=triviaqa
run_task

TASK_NAME=squadv2
run_task

TASK_NAME=commonsense_qa
run_task

TASK_NAME=bertaqa_en
run_task