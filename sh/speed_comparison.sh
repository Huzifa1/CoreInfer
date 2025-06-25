#!/bin/sh

# SIOT Method Variables: 
START_NUM=4
END_NUM=26
BASE_NEURONS_PERCENT=0.3
BASE_NEURONS_TYPE="dataset"
LOADED_NEURONS_PERCENT=0.7
MODEL_NEURONS_FILEPATH="neurons/llama3-3b_model_neurons.json"
DATASET_NEURONS_FILEPATH="neurons/qa.json"
MASK_FILEPATH="neurons/mask.pkl"


DIR_PATH="$(cd .. && pwd)"
TOKEN_SPARSITY=0.2
SPARSITY=0.4
LIMIT=100
USE_SIOT_IMPROVEMENTS=True
METHOD="siot"
TASK_NAME="triviaqa"
MODEL_NAME="llama3-3b"

NUM_TOKEN_GENERATE=50

run()
{
    echo "USE_SIOT_IMPROVEMENTS = $USE_SIOT_IMPROVEMENTS" > $DIR_PATH/transformers/siot_variables/siot_improvements.py
    OUT_DIR="$DIR_PATH/results/speed_comparison/$TASK_NAME/results[$LIMIT]"
    mkdir -p $OUT_DIR
    echo "Output path: $OUT_DIR, file: $FILE_NAME"
    python coreinfer-dataset.py --model_name $MODEL_NAME --num_tokens_to_generate $NUM_TOKEN_GENERATE --task_type 'QA' --checkpoint_path './models/llama3-3b/' --sparsity $SPARSITY --base_neurons_percent $BASE_NEURONS_PERCENT --dataset_neurons_filepath $DATASET_NEURONS_FILEPATH --loaded_neurons_percent $LOADED_NEURONS_PERCENT --method $METHOD --max_items $LIMIT --output_path "$OUT_DIR/${FILE_NAME}"
}

cd ..
for TASK_NAME in "triviaqa" "squadv2" "wmt16-de-en" "wmt16-ro-en" "xsum" "cnn_dailymail"; do

    if [ "$TASK_NAME" = "triviaqa" ] || [ "$TASK_NAME" = "squadv2" ]; then
        DATASET_NEURONS_FILEPATH="neurons/qa.json"
    elif [ "$TASK_NAME" = "wmt16-de-en" ] || [ "$TASK_NAME" = "wmt16-ro-en" ]; then
        DATASET_NEURONS_FILEPATH="neurons/translate.json"
    elif [ "$TASK_NAME" = "xsum" ] || [ "$TASK_NAME" = "cnn_dailymail" ]; then
        DATASET_NEURONS_FILEPATH="neurons/summarize.json"
    fi

    LOADED_NEURONS_PERCENT=0.7
    METHOD="siot"
    FILE_NAME="siot_0.7.txt"
    run

    LOADED_NEURONS_PERCENT=0.7
    METHOD="dense"
    FILE_NAME="dense_0.7.txt"
    run

    LOADED_NEURONS_PERCENT=0.4
    METHOD="dense"
    FILE_NAME="dense_0.4.txt"
    run
done

