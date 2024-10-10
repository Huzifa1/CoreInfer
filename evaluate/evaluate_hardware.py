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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer

    
# Convert Model
def convert_model(method, model, model_name, sparsity, token_sparsity, memory_limit, cluster_path=None):
    if "opt" in model_name:
        if method == 'stable_guided':
            model = convert_opt_model(model, sparsity, token_sparsity, memory_limit)
        elif method == 'similarity_guided':
            model = convert_opt_model_sim(model, num_layers, sparsity, memory_limit, cluster_path)
            
        
    elif "llama" in model_name:
        if method == 'stable_guided':
            model = convert_llama_model(model, sparsity, token_sparsity, memory_limit)
        elif method == 'similarity_guided':
            model = convert_llama_model_sim(model, num_layers, sparsity, memory_limit, cluster_path)
        
    return model




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



def main(method, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, num_tokens_to_generate, memory_limit, device, cluster_path = None):
    model, tokenizer = _load_model(checkpoint_path, start_num, end_num, device, memory_limit)
    
    if cluster_path is None:
        model  = convert_model(method, model, model_name, sparsity, start_num, end_num, token_sparsity, memory_limit)
    else:
        model  = convert_model(method, model, model_name, sparsity, start_num, end_num, token_sparsity, memory_limit, cluster_path)
        
    evaluate(model, tokenizer, num_tokens_to_generate, device)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="opt-6.7b", help='Model Name')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--sparsity', type=float, default=0, help='Sparsity level.')
    parser.add_argument('--start_num', type=int, default=5, help='Start layer.')
    parser.add_argument('--end_num', type=int, default=27, help='End layer.')
    parser.add_argument('--token_sparsity', type=float, default=0.2, help='Token Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')
    parser.add_argument('--method', type=str, choices=['stable_guided', 'similarity_guided'], default='stable_guided', help='Method to use (default: stable_guided).')
    parser.add_argument('--cluster_path', type=str, default=None, help='Optional cluster path.')
    parser.add_argument('--num_tokens_to_generate', type=int, default=256, help='Maximum number of new tokens.')

    args = parser.parse_args()
    
    main(args.method, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity, args.num_tokens_to_generate, 
         args.memory_limit, args.device, args.cluster_path)


    