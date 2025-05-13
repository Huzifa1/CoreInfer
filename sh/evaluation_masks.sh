
DATE='2025_05_13_14_57'
TASK_NAME=wmdp_cyber
LIMIT=100


run()
{
    echo 'MASK_FILEPATH = "masks/scores_'$DATE'_'$TASK_NAME'_'$SPARSITY'_0.7_4_26.mask"' > transformers/mask_filepath.py
    python evaluation/evaluate_task.py --model_name llama3-3b --num_fewshot 0 --checkpoint_path models/llama3-3b/ --cpu_only --limit $LIMIT --task_name $TASK_NAME --method dense --output_path 'results/dataset_run_'$DATE'_'$TASK_NAME'_dense_'$SPARSITY'.json'
}

cd ..
SPARSITY=0.1
run
SPARSITY=0.15
run
SPARSITY=0.2
run
SPARSITY=0.25
run
SPARSITY=0.3
run
