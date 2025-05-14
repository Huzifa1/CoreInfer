#!/bin/sh

# init the variables
DATE=0
TASK_NAME=0
LIMIT=0
NUM_FEWSHOT=3

# set the variables
. ./variables.sh

run()
{
    echo 'USE_SIOT_IMPROVEMENTS = True' > transformers/siot_variables/siot_improvements.py
    echo 'MASK_FILEPATH = "masks/scores_'$DATE'_'$TASK_NAME'_'$SPARSITY'_0.7_4_26.mask"' > transformers/siot_variables/mask_filepath.py
    python evaluation/evaluate_task.py --model_name llama3-3b --num_fewshot $NUM_FEWSHOT --checkpoint_path models/llama3-3b/ --cpu_only --limit $LIMIT --task_name $TASK_NAME --method dense --output_path 'results/dataset_run_'$DATE'_'$TASK_NAME'_dense_'$SPARSITY'.json'
}

cd ..
SPARSITY=0.25
run
SPARSITY=0.3
run
SPARSITY=0.35
run
SPARSITY=0.4
run
