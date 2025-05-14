
TASK_NAME='triviaqa'
LIMIT=800
NUM_FEWSHOT=3


cd ..
echo 'USE_SIOT_IMPROVEMENTS = False' > transformers/siot_variables/siot_improvements.py
python evaluation/evaluate_task.py --model_name llama3-3b --num_fewshot $NUM_FEWSHOT --checkpoint_path models/llama3-3b/ --cpu_only --limit $LIMIT --task_name $TASK_NAME --method score