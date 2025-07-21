import torch
import time
from pathlib import Path
from transformers import OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from common import *
from transformers.siot import USE_SIOT_IMPROVEMENTS
import create_neurons_mask

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Test Model
def evaluate(model, tokenizer, num_tokens_to_generate, device):
    prompt = "Once a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        prefill_output = model.generate(
            input_ids,
            max_length=input_ids.size(1) + 1,  # 只生成一个 token
            do_sample=False,
            # use_cache=True
        )
    
    input_ids_prefilled = prefill_output[:, :-1]
    eos_token_id = tokenizer.convert_tokens_to_ids('.')
    
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            input_ids_prefilled,
            max_length=input_ids_prefilled.size(1) + num_tokens_to_generate,
            do_sample=False, 
            # use_cache=True
        )
        
    end_time = time.time()
    num_generated_tokens = output.size(1) - input_ids_prefilled.size(1)

    elapsed_time = end_time - start_time
    tokens_per_second = num_generated_tokens / elapsed_time
    
    print(f'Generated {num_generated_tokens} tokens in {elapsed_time:.2f} seconds.')
    print(f'Decoding speed: {tokens_per_second:.2f} tokens/second')


def main(method, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, prompt, memory_limit, num_fewshot, task_type, num_tokens_to_generate, device, sampling_method, siot_method_config, cluster_path = None, cpu_only = False, top_p = None, sparsity_levels_path = None, hybrid_split = None, model_neurons_filepath = None):
    
    if (USE_SIOT_IMPROVEMENTS and method == "siot"):
        create_neurons_mask.main(start_num, end_num, siot_method_config)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(f"{script_dir}/transformers/siot_variables/mask_filepath.txt", "r") as f:
            MASK_FILEPATH = f.readlines()[0]
        print(f"SIOT: Use Mask for partial loading, mask file: {MASK_FILEPATH}") 
    
    model, tokenizer, num_layers = load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit)
    
    model = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, siot_method_config, cluster_path, cpu_only, sparsity_levels_path, hybrid_split, model_neurons_filepath)

    evaluate(model, tokenizer, num_tokens_to_generate, device)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Llama-3.2-3B", help='Model Name')
    parser.add_argument('--prompt', type=str, default="What happens to you if you eat watermelon seeds?", help='Input prompt.')
    parser.add_argument('--num_fewshot', type=int, default=6, help='Number of samples.')
    parser.add_argument('--num_tokens_to_generate', type=int, default=256, help='Maximum number of new tokens.')
    parser.add_argument('--task_type', type=str, choices=['QA', 'Summarize', 'translate_de_en'], default='QA', help='Type of task to perform.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("meta-llama/Llama-3.2-3B"), help='Model checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--sparsity', type=float, default=0.4, help='Sentence Sparsity level.')
    parser.add_argument('--start_num', type=int, default=3, help='Start layer.')
    parser.add_argument('--end_num', type=int, default=25, help='End layer.')
    parser.add_argument('--top_p', type=float, default=0.9, help='When set, will use top-p sampling')
    parser.add_argument('--sampling-method', type=str, default="greedy", choices=["greedy", "top-p"], help='Choose sampling method')
    parser.add_argument('--token_sparsity', type=float, default=0.2, help='Token Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')
    parser.add_argument('--method', type=str, choices=['stable_guided', 'similarity_guided', 'dynamic_cut', 'dense', 'static_cut', 'moving_cut', 'sparsity_levels', 'siot', 'model_neurons', 'hybrid_neurons'], default='stable_guided', help='Method to use (default: stable_guided).')
    parser.add_argument('--cluster_path', type=str, default=None, help='Optional cluster path.')
    parser.add_argument('--cpu_only', action='store_true', help='Run inference on CPU only.')
    parser.add_argument('--sparsity_levels_path', type=Path, default=None, help='Path to sparsity levels file.')
    parser.add_argument('--hybrid_split', type=float, default=0.5, help='Amout of model neurons')
    
    ## SIOT Method arguments
    parser.add_argument('--base_neurons_percent', type=float, default=0.3, help='Loaded Base Neurons Percent')
    parser.add_argument('--base_neurons_type', type=str, choices=['model', 'dataset'], default='dataset', help='Base Neurons Type')
    parser.add_argument('--loaded_neurons_percent', type=float, default=0.7, help='Overall Percent of Loaded Neurons')
    parser.add_argument('--model_neurons_filepath', type=Path, default="neurons/llama3-3b_new_model_neurons.json", help='Path to model neurons file')
    parser.add_argument('--dataset_neurons_filepath', type=Path, default="neurons/qa.json", help='Path to dataset neurons file')
    parser.add_argument('--mask_filepath', type=Path, default="neurons/mask.pkl", help='Path to output mask file')


    args = parser.parse_args()
    if (args.cpu_only):
        default_device = 'cpu'
        args.device = 'cpu'
    if args.cpu_only and args.memory_limit:
        parser.error("The options --cpu_only and --memory_limit cannot be used together.")
  
    if args.method == 'sparsity_levels' and args.sparsity_levels_path is None:
        parser.error("The option --sparsity_levels_path is required when using the sparsity_levels method.")
        
    if (args.method == "model_neurons" or args.method == "hybrid_neurons") and args.model_neurons_filepath is None:
        parser.error(f"The option --model_neurons_filepath is required when using the {args.method} method.")
        
    siot_method_config = {
        "base_neurons_percent": args.base_neurons_percent,
        "base_neurons_type": args.base_neurons_type,
        "loaded_neurons_percent": args.loaded_neurons_percent,
        "model_neurons_filepath": args.model_neurons_filepath,
        "dataset_neurons_filepath": args.dataset_neurons_filepath,
        "mask_filepath": args.mask_filepath
    }

    main(args.method, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity, args.prompt, args.memory_limit, args.num_fewshot, args.task_type, args.num_tokens_to_generate, args.device, args.sampling_method, siot_method_config, args.cluster_path, args.cpu_only, args.top_p, args.sparsity_levels_path, args.hybrid_split, args.model_neurons_filepath)
