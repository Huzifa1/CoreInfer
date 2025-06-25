import torch
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from common import *
import json
from torch.nn.functional import softmax
from transformers.siot import USE_SIOT_IMPROVEMENTS
import create_neurons_mask

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
MASK_FILEPATH = ""
SHOW_DECODING_SPEED = True


# To encode Path variables into json files
class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)
    
def top_p_sampling(next_token_logits, tokenizer, top_p, generated):
    # Apply softmax to get probabilities
    probabilities = softmax(next_token_logits, dim=-1)

    # Sort tokens by probability
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep only tokens within the top-p threshold
    top_p_mask = cumulative_probs <= top_p
    top_p_mask[..., 0] = True  # Ensure at least one token is always included

    # Re-normalize the probabilities of the selected tokens
    filtered_probs = sorted_probs * top_p_mask
    filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Avoid division by zero

    # Sample from the filtered distribution
    next_token_id = sorted_indices.gather(1, torch.multinomial(filtered_probs, num_samples=1))

    # Ensure shape is [1, 1]
    next_token_id = next_token_id.squeeze(-1).unsqueeze(1)

    # Concatenate to generated sequence
    generated = torch.cat((generated, next_token_id), dim=1)

    next_token_text = tokenizer.decode(next_token_id.squeeze().tolist())    

    return generated, next_token_id, next_token_text

def chat(model, tokenizer, num_fewshot, limit, config, device):
    model.eval()
    
    prompt_counter = 0
    while True:
        user_prompt = input("You: ")
        
        if (user_prompt == "exit"):
            break
        
        prompt = process_prompt_stable(user_prompt, "QA", num_fewshot)
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_token_length = input_ids.shape[-1]
        eos_token_id = tokenizer.convert_tokens_to_ids('.')
        
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
        
        start_time = time.time()
        answer = ""
        top_p = 0.9
        generated = input_ids
        num_tokens_to_generate = limit
        token_counter = 0
        for _ in range(num_tokens_to_generate):
            with torch.no_grad():
                outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits
                past_key_values = outputs.past_key_values
        
            next_token_logits = logits[:, -1, :]

            generated, next_token_id, next_token_text = top_p_sampling(next_token_logits, tokenizer, top_p, generated)
            
            print(next_token_text, end='', flush=True)
            answer += next_token_text
            
            if next_token_id.item() == eos_token_id:
                break
            if '.' in next_token_text or "\n" in next_token_text:
                break
            if token_counter >= num_tokens_to_generate:
                break
            token_counter += 1
        
        if SHOW_DECODING_SPEED:
            end_time = time.time()
            
            num_generated_tokens = generated.size(1) - input_ids.size(1)
            
            elapsed_time = end_time - start_time
            tokens_per_second = num_generated_tokens / elapsed_time
            print(f"Decoding speed: {token_counter} token/sec\n")
        
        answer = answer.replace("\n", "")
        print(answer)
        
        print("\n\n")
        prompt_counter += 1


def main(method, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, memory_limit, device, num_fewshot, limit, siot_method_config, num_tokens_to_generate, cluster_path = None, cpu_only = None, hybrid_split = None, model_neurons_filepath = None):
    
    if (USE_SIOT_IMPROVEMENTS and method == "siot"):
        create_neurons_mask.main(start_num, end_num, siot_method_config)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        with open(f"{parent_dir}/transformers/siot_variables/mask_filepath.txt", "r") as f:
            MASK_FILEPATH = f.readlines()[0]
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
    chat(model, tokenizer, num_fewshot, limit, config, device)
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="opt-6.7b", help='Model Name')
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
    parser.add_argument('--cpu_only', action='store_true', help='Run inference on CPU only.')
    parser.add_argument('--hybrid_split', type=float, default=0.5, help='Amout of model neurons')
    
    ## SIOT Method arguments
    parser.add_argument('--base_neurons_percent', type=float, default=0.4, help='Loaded Base Neurons Percent')
    parser.add_argument('--base_neurons_type', type=str, choices=['model', 'dataset'], default='model', help='Base Neurons Type')
    parser.add_argument('--loaded_neurons_percent', type=float, default=0.7, help='Overall Percent of Loaded Neurons')
    parser.add_argument('--model_neurons_filepath', type=Path, default="neurons/llama3-3b_model_neurons.json", help='Path to model neurons file')
    parser.add_argument('--dataset_neurons_filepath', type=Path, default="neurons/qa.json", help='Path to dataset neurons file')
    parser.add_argument('--mask_filepath', type=Path, default="neurons/mask.pkl", help='Path to output mask file')
    
    


    args = parser.parse_args()

    if (args.cpu_only):
        default_device = 'cpu'
        args.device = 'cpu'

    if args.cpu_only and args.memory_limit:
        parser.error("The options --cpu_only and --memory_limit cannot be used together.")
        
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
    
    main(args.method, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity,
         args.memory_limit, args.device, args.num_fewshot, args.limit, siot_method_config, args.cluster_path, args.cpu_only, args.hybrid_split, args.model_neurons_filepath)
