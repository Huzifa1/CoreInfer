#!/bin/sh
DIR_PATH="$(cd .. && pwd)"

# Tunable Params
METHOD="stable_guided"
MODEL_NAME="llama3-3b"
LIMIT=10000000
FILE_PREFIX_NAME="coreinfer_random_loading"
USE_SIOT_IMPROVEMENTS=True
TOKEN_SPARSITY=0.2
SPARSITY=0.4

# SIOT Method Variables: 
START_NUM=4
END_NUM=26
BASE_NEURONS_PERCENT=0.7
BASE_NEURONS_TYPE="model"
LOADED_NEURONS_PERCENT=0.7
MODEL_NEURONS_FILEPATH="neurons_files/${MODEL_NAME}/random_neurons.json"
DATASET_NEURONS_FILEPATH="neurons_files/${MODEL_NAME}/qa.json"
MASK_FILEPATH="neurons_files/mask2.pkl"

run()
{
    echo "USE_SIOT_IMPROVEMENTS = $USE_SIOT_IMPROVEMENTS" > $DIR_PATH/transformers/siot_variables/siot_improvements.py
    cd $DIR_PATH/evaluation
    OUT_DIR="$DIR_PATH/results/${METHOD}_results/$TASK_NAME/results[$LIMIT]"
    mkdir -p $OUT_DIR
    python evaluate_task.py --model_name $MODEL_NAME --checkpoint_path /local/huzaifa/workspace/models/$MODEL_NAME/ --limit $LIMIT --task_name $TASK_NAME --output_path "$OUT_DIR/${FILE_NAME}" --token_sparsity $TOKEN_SPARSITY --sparsity $SPARSITY --method $METHOD --start_num $START_NUM --end_num $END_NUM --base_neurons_percent $BASE_NEURONS_PERCENT --base_neurons_type $BASE_NEURONS_TYPE --loaded_neurons_percent $LOADED_NEURONS_PERCENT --model_neurons_filepath $MODEL_NEURONS_FILEPATH --dataset_neurons_filepath $DATASET_NEURONS_FILEPATH --mask_filepath $MASK_FILEPATH
    cd -
}


for TASK_NAME in mlqa_en_en; do # "triviaqa" "squadv2" "wmt16-de-en" "wmt16-ro-en" "xsum" "cnn_dailymail"

    if [ "$TASK_NAME" = "triviaqa" ] || [ "$TASK_NAME" = "squadv2" ] || [ "$TASK_NAME" = "piqa" ] || [ "$TASK_NAME" = "mlqa_en_en" ]; then
        DATASET_NEURONS_FILEPATH="neurons_files/$MODEL_NAME/qa.json"
    elif [ "$TASK_NAME" = "wmt16-de-en" ] || [ "$TASK_NAME" = "wmt16-ro-en" ]; then
        DATASET_NEURONS_FILEPATH="neurons_files/$MODEL_NAME/translate.json"
    elif [ "$TASK_NAME" = "xsum" ] || [ "$TASK_NAME" = "cnn_dailymail" ]; then
        DATASET_NEURONS_FILEPATH="neurons_files/$MODEL_NAME/summarize.json"
    fi

    FILE_NAME="${FILE_PREFIX_NAME}_${MODEL_NAME}.json"
    run
done

