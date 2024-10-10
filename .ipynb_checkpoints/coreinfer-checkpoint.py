import torch
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *
from convert.convert_opt_model import convert_opt_model
from convert.convert_opt_model_sim import convert_opt_model_sim
from convert.convert_llama_model import convert_llama_model
from convert.convert_llama_model_sim import convert_llama_model_sim


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
def _load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit):
    if memory_limit == True:
        if "opt" in model_name:
            model, num_layers = load_opt_model(checkpoint_path, start_num, end_num)
                    
        elif "llama" in model_name:
            model, num_layers = load_llama_model(checkpoint_path, start_num, end_num)
                
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map=device, torch_dtype=torch.float16)
        num_layers = model.config.num_hidden_layers
        
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, device_map=device)
        
    return model, tokenizer, num_layers

    
# Convert Model
def convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, cluster_path=None):
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
def generate(method, model, tokenizer, ori_prompt, task_type, num_fewshot, num_tokens_to_generate, device):
    model.eval()
    if method == 'stable_guided':
        prompt = process_prompt_stable(ori_prompt, task_type, num_fewshot)
    elif method == 'similarity_guided':
        prompt = process_prompt_similarity(ori_prompt, task_type)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)


    eos_token_id = tokenizer.convert_tokens_to_ids('.')
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
    
    generated = input_ids
    
    print(ori_prompt) 
    start_time = time.time()
    
    for _ in range(num_tokens_to_generate):
        with torch.no_grad():
            outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
    
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
    
        generated = torch.cat((generated, next_token_id.unsqueeze(-1)), dim=1)
    
        next_token_text = tokenizer.decode(next_token_id)
        print(next_token_text, end='', flush=True)
    
        if next_token_id.item() == eos_token_id:
            break
        if '.' in next_token_text:
            break
    
    end_time = time.time()
    
    num_generated_tokens = generated.size(1) - input_ids.size(1)
    
    elapsed_time = end_time - start_time
    tokens_per_second = num_generated_tokens / elapsed_time
    
    print(f'\nGenerated {num_generated_tokens} tokens in {elapsed_time:.2f} seconds.')
    print(f'Decoding speed: {tokens_per_second:.2f} tokens/second')





def main(method, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, prompt, memory_limit, num_fewshot, task_type, num_tokens_to_generate, device, cluster_path = None):
    model, tokenizer, num_layers = _load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit)
    
    if cluster_path is None:
        model  = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit)
    else:
        model  = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, cluster_path)
        
    generate(method, model, tokenizer, prompt, task_type, num_fewshot, num_tokens_to_generate, device)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="opt-6.7b", help='Model Name')
    parser.add_argument('--prompt', type=str, default="What happens to you if you eat watermelon seeds?", help='Input prompt.')
    parser.add_argument('--num_fewshot', type=int, default=6, help='Number of samples.')
    parser.add_argument('--num_tokens_to_generate', type=int, default=256, help='Maximum number of new tokens.')
    parser.add_argument('--task_type', type=str, choices=['QA', 'Summarize', 'translate_de_en'], default='QA', help='Type of task to perform.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--sparsity', type=float, default=0.4, help='Sentence Sparsity level.')
    parser.add_argument('--start_num', type=int, default=5, help='Start layer.')
    parser.add_argument('--end_num', type=int, default=27, help='End layer.')
    parser.add_argument('--token_sparsity', type=float, default=0.2, help='Token Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')
    parser.add_argument('--method', type=str, choices=['stable_guided', 'similarity_guided'], default='stable_guided', help='Method to use (default: stable_guided).')
    parser.add_argument('--cluster_path', type=str, default=None, help='Optional cluster path.')

    args = parser.parse_args()

    main(args.method, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity, args.prompt, args.memory_limit,
        args.num_fewshot, args.task_type, args.num_tokens_to_generate, args.device, args.cluster_path)








