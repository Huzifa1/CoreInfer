#!/bin/sh

# SIOT Method Variables: 
START_NUM=4
END_NUM=26
BASE_NEURONS_PERCENT=0.3
BASE_NEURONS_TYPE="dataset"
LOADED_NEURONS_PERCENT=0.7
MODEL_NEURONS_FILEPATH="neurons/llama3-3b_model_neurons_new.json"
DATASET_NEURONS_FILEPATH="neurons/qa.json"
MASK_FILEPATH="neurons/mask.pkl"


DIR_PATH="$(cd .. && pwd)"
TOKEN_SPARSITY=0.2
SPARSITY=0.4
LIMIT=500
USE_SIOT_IMPROVEMENTS=True
METHOD="siot"
TASK_NAME="triviaqa"
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


for TASK_NAME in "triviaqa" "squadv2" "wmt16-de-en" "wmt16-ro-en" "xsum" "cnn_dailymail"; do

    if [ "$TASK_NAME" = "triviaqa" ] || [ "$TASK_NAME" = "squadv2" ]; then
        DATASET_NEURONS_FILEPATH="neurons/qa.json"
    elif [ "$TASK_NAME" = "wmt16-de-en" ] || [ "$TASK_NAME" = "wmt16-ro-en" ]; then
        DATASET_NEURONS_FILEPATH="neurons/translate.json"
    elif [ "$TASK_NAME" = "xsum" ] || [ "$TASK_NAME" = "cnn_dailymail" ]; then
        DATASET_NEURONS_FILEPATH="neurons/summarize.json"
    fi

    FILE_NAME="task_specific.json"
    run
done

