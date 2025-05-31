#!/bin/sh

# SIOT Method Variables: 
START_NUM=5
END_NUM=27
BASE_NEURONS_PERCENT=0.4
BASE_NEURONS_TYPE="model"
LOADED_NEURONS_PERCENT=0.7
MODEL_NEURONS_FILEPATH="neurons/llama3-3b_model_neurons.json"
DATASET_NEURONS_FILEPATH="neurons/truthfulqa_gen_dataset_neurons.json"
MASK_FILEPATH="neurons/mask.pkl"


DIR_PATH="$(cd .. && pwd)"
TOKEN_SPARSITY=0.2
SPARSITY=0.4
LIMIT=1
USE_SIOT_IMPROVEMENTS=True
METHOD="siot"
TASK_NAME="truthfulqa_gen"
MODEL_NAME="llama3-3b"

run()
{
    echo "USE_SIOT_IMPROVEMENTS = $USE_SIOT_IMPROVEMENTS" > $DIR_PATH/transformers/siot_variables/siot_improvements.py
    cd $DIR_PATH/evaluation
    OUT_DIR="$DIR_PATH/${METHOD}_results/$TASK_NAME/results[$LIMIT]"
    mkdir -p $OUT_DIR
    python evaluate_task.py --model_name $MODEL_NAME --checkpoint_path $DIR_PATH/models/$MODEL_NAME/ --cpu_only --limit $LIMIT --task_name $TASK_NAME --output_path "$OUT_DIR/${FILE_NAME}" --token_sparsity $TOKEN_SPARSITY --sparsity $SPARSITY --method $METHOD --start_num $START_NUM --end_num $END_NUM --base_neurons_percent $BASE_NEURONS_PERCENT --base_neurons_type $BASE_NEURONS_TYPE --loaded_neurons_percent $LOADED_NEURONS_PERCENT --model_neurons_filepath $MODEL_NEURONS_FILEPATH --dataset_neurons_filepath $DATASET_NEURONS_FILEPATH --mask_filepath $MASK_FILEPATH
    cd -
}


for TASK_NAME in "truthfulqa_gen" "triviaqa" "wmt16-de-en" "squadv2"; do
    DATASET_NEURONS_FILEPATH="neurons/${$TASK_NAME}_dataset_neurons.json"
    # Make sure to include the variable you are tuning in the filename, so that files don't overlap
    # All information are included in the evaluation result file.
    # If you want to tune another variable, I suggest to use a different directory (EDIT $OUT_DIR)

    # For example, if you are tuning BASE_NEURON_PERCENT:
    for BASE_NEURON_PERCENT in 0.1 0.2 0.3 0.4; do
        FILE_NAME="${BASE_NEURON_PERCENT}.json"
        run
    done
done

