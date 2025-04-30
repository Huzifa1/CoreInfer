import torch
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from common import *
import json

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Test Model
import lm_eval
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

def evaluate(task_name, model, tokenizer, num_fewshot, device, limit, output_path):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model = hflm,
        tasks = [task_name],
        num_fewshot=num_fewshot,
        limit = limit,
        simple_evaluate=True,
        random_seed=42,
        write_out=True
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
        json.dump(results['results'], file)



def main(method, task_name, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, memory_limit, device, num_fewshot, limit, output_path, cluster_path = None, cpu_only = None):
    
    model, tokenizer, num_layers = load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit)
    
    model = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, cluster_path, cpu_only)
        
    evaluate(task_name, model, tokenizer, num_fewshot, device, limit, output_path)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="opt-6.7b", help='Model Name')
    parser.add_argument('--task_name', type=str, default="truthfulqa_gen", help='Task Name')
    parser.add_argument('--num_fewshot', type=int, default=6, help='Number of samples.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--sparsity', type=float, default=0, help='Sparsity level.')
    parser.add_argument('--start_num', type=int, default=5, help='Start layer.')
    parser.add_argument('--end_num', type=int, default=27, help='End layer.')
    parser.add_argument('--limit', type=int, default=1000, help='Max number of samples to evaluate.')
    parser.add_argument('--token_sparsity', type=float, default=0.2, help='Token Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')
    parser.add_argument('--method', type=str, choices=['stable_guided', 'similarity_guided', 'dynamic_cut', 'dynamic_cut_ci', 'dense', 'static_cut', 'moving_cut', 'sparsity_levels', 'score'], default='stable_guided', help='Method to use (default: stable_guided).')
    parser.add_argument('--cluster_path', type=str, default=None, help='Optional cluster path.')
    parser.add_argument('--output_path', type=Path, default=None, help='Path to output file.')
    parser.add_argument('--cpu_only', action='store_true', help='Run inference on CPU only.')

    args = parser.parse_args()

    if (args.cpu_only):
        default_device = 'cpu'
        args.device = 'cpu'

    if args.cpu_only and args.memory_limit:
        parser.error("The options --cpu_only and --memory_limit cannot be used together.")
    
    main(args.method, args.task_name, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity,
         args.memory_limit, args.device, args.num_fewshot, args.limit, args.output_path, args.cluster_path, args.cpu_only)


    