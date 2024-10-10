import torch
import time
from pathlib import Path
from transformers import OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from convert.convert_opt_model import convert_opt_model
from convert.convert_opt_model_sim import convert_opt_model_sim
from convert.convert_llama_model import convert_llama_model
from convert.convert_llama_model_sim import convert_llama_model_sim

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
def _load_model(checkpoint_path, start_num, end_num, device, memory_limit):
    if memory_limit == True:
        if "opt" in model_name:
            model, num_layers = load_opt_model(checkpoint_path, start_num, end_num)
                    
        elif "llama" in model_name:
            model, num_layers = load_llama_model(checkpoint_path, start_num, end_num)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map=device, torch_dtype=torch.float16)
        
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, device_map=device)
        
    return model, tokenizer

    
# Convert Model
def convert_model(method, model, model_name, sparsity, start_num, end_num, token_sparsity, memory_limit, cluster_path=None):
    if "opt" in model_name:
        if method == 'stable_guided':
            model = convert_opt_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit)
        elif method == 'similarity_guided':
            model = convert_opt_model_sim(model, num_layers, sparsity, start_num, end_num, memory_limit, cluster_path)
            
        
    elif "llama" in model_name:
        if method == 'stable_guided':
            model = convert_llama_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit)
        elif method == 'similarity_guided':
            model = convert_llama_model_sim(model, num_layers, sparsity, start_num, end_num, memory_limit, cluster_path)
        
    return model




# Test Model
import lm_eval
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

def evaluate(task_name, model, tokenizer, num_fewshot, device):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
    model=hflm,
    tasks=[task_name],
    num_fewshot=num_fewshot)
    print(results['results'])



def main(method, task_name, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, memory_limit, device, num_fewshot, cluster_path = None):
    model, tokenizer = _load_model(checkpoint_path, start_num, end_num, device, memory_limit)
    
    if cluster_path is None:
        model  = convert_model(method, model, model_name, sparsity, start_num, end_num, token_sparsity, memory_limit)
    else:
        model  = convert_model(method, model, model_name, sparsity, start_num, end_num, token_sparsity, memory_limit, cluster_path)
        
    evaluate(task_name, model, tokenizer, num_fewshot, device)





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
    parser.add_argument('--token_sparsity', type=float, default=0.2, help='Token Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')
    parser.add_argument('--method', type=str, choices=['stable_guided', 'similarity_guided'], default='stable_guided', help='Method to use (default: stable_guided).')
    parser.add_argument('--cluster_path', type=str, default=None, help='Optional cluster path.')

    args = parser.parse_args()
    
    main(args.method, args.task_name, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity,
         args.memory_limit, args.device, args.num_fewshot, args.cluster_path)


    