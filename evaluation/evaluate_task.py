import torch
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from common import *
import json
from scores.neuron_score_writer import write_neuron_scores
from transformers.siot import USE_SIOT_IMPROVEMENTS, MASK_FILEPATH
from convert.convert_llama_model_score import SPARSITY_LEVELS
import create_neurons_mask

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Test Model
import lm_eval
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

# To encode Path variables into json files
class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def evaluate(task_name, model, tokenizer, num_fewshot, limit, output_path, config):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model = hflm,
        tasks = [task_name],
        num_fewshot=num_fewshot,
        limit = limit,
        log_samples=True,
        random_seed=42,
        write_out=True
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_dict = results['results']
    if (USE_SIOT_IMPROVEMENTS):
        result_dict["mask_name"] = MASK_FILEPATH
        result_dict["config"] = config
    command_str = f"Command: {' '.join(sys.argv)}"
    result_dict['command'] = command_str
    with open(output_path, 'w') as file:
        json.dump(result_dict, file, cls=PathEncoder)



def main(method, task_name, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, memory_limit, device, num_fewshot, limit, output_path, siot_method_config, cluster_path = None, cpu_only = None, hybrid_split = None, model_neurons_filepath = None):
    
    if (USE_SIOT_IMPROVEMENTS and method == "siot"):
        create_neurons_mask.main(start_num, end_num, siot_method_config)
        print(f"SIOT: Use Mask for partial loading, mask file: {MASK_FILEPATH}") 
    
    model, tokenizer, num_layers = load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit)
    
    model = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, siot_method_config, cluster_path, cpu_only, hybrid_split=hybrid_split, model_neurons_filepath=model_neurons_filepath)
        
    config = {
        "sparsity": sparsity,
        "start_num": start_num,
        "end_num": end_num,
        "token_sparsity": token_sparsity,
        "siot_method_config": siot_method_config
    }
    evaluate(task_name, model, tokenizer, num_fewshot, limit, output_path, config)
    


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
    parser.add_argument('--method', type=str, choices=['stable_guided', 'similarity_guided', 'dynamic_cut', 'dynamic_cut_ci', 'dense', 'static_cut', 'moving_cut', 'sparsity_levels', 'score', 'siot', 'model_neurons', 'hybrid_neurons'], default='stable_guided', help='Method to use (default: stable_guided).')
    parser.add_argument('--cluster_path', type=str, default=None, help='Optional cluster path.')
    parser.add_argument('--output_path', type=Path, default=None, help='Path to output file.')
    parser.add_argument('--cpu_only', action='store_true', help='Run inference on CPU only.')
    parser.add_argument('--hybrid_split', type=float, default=0.5, help='Amout of model neurons')
    
    ## SIOT Method arguments
    parser.add_argument('--base_neurons_percent', type=float, default=0.4, help='Loaded Base Neurons Percent')
    parser.add_argument('--base_neurons_type', type=str, choices=['model', 'dataset'], default='model', help='Base Neurons Type')
    parser.add_argument('--loaded_neurons_percent', type=float, default=0.7, help='Overall Percent of Loaded Neurons')
    parser.add_argument('--model_neurons_filepath', type=Path, default="neurons/llama3-3b_model_neurons.json", help='Path to model neurons file')
    parser.add_argument('--dataset_neurons_filepath', type=Path, default="neurons/truthfulqa_gen_dataset_neurons.json", help='Path to dataset neurons file')
    parser.add_argument('--mask_filepath', type=Path, default="neurons/mask.pkl", help='Path to output mask file')
    
    


    args = parser.parse_args()

    if (args.cpu_only):
        default_device = 'cpu'
        args.device = 'cpu'

    if args.cpu_only and args.memory_limit:
        parser.error("The options --cpu_only and --memory_limit cannot be used together.")
        
    if (args.output_path == None):
        timestr = time.strftime("%Y_%m_%d_%H_%M")
        args.output_path = f"results/dataset_run_{timestr}_{args.task_name}_{args.method}.json"
        
    if (args.method == "model_neurons" or args.method == "hybrid_neurons") and args.model_neurons_filepath is None:
        parser.error(f"The option --model_neurons_filepath is required when using the {args.method} method.")
        
    print(f"Use filename {args.output_path}\n")
    
    
    siot_method_config = {
        "base_neurons_percent": args.base_neurons_percent,
        "base_neurons_type": args.base_neurons_type,
        "loaded_neurons_percent": args.loaded_neurons_percent,
        "model_neurons_filepath": args.model_neurons_filepath,
        "dataset_neurons_filepath": args.dataset_neurons_filepath,
        "mask_filepath": args.mask_filepath
    }
    
    main(args.method, args.task_name, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity,
         args.memory_limit, args.device, args.num_fewshot, args.limit, args.output_path, siot_method_config, args.cluster_path, args.cpu_only, args.hybrid_split, args.model_neurons_filepath)


    