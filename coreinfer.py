import torch
import time
from pathlib import Path
from utils import *
from common import *
from torch.nn.functional import softmax
from transformers.siot import USE_SIOT_IMPROVEMENTS
import create_neurons_mask

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def greedy_sampling(next_token_logits, tokenizer, generated):
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    generated = torch.cat((generated, next_token_id.unsqueeze(-1)), dim=1)
    next_token_text = tokenizer.decode(next_token_id)

    return generated, next_token_id, next_token_text

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

# Test Model
def generate(method, model, tokenizer, ori_prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p, show_debug: bool = True):
    model.eval()
    if method in ['stable_guided', 'static_cut', 'dynamic_cut', 'dense', 'dynamic_cut_ci', 'model_neurons', 'hybrid_neurons', 'siot']:
        prompt = process_prompt_stable(ori_prompt, task_type, num_fewshot)
    elif method == 'similarity_guided':
        prompt = process_prompt_similarity(ori_prompt, task_type)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_token_length = input_ids.shape[-1]
    if show_debug:
        print(f"Prompt token length: {prompt_token_length}")
    pre_fill_start_time = time.time()
    if show_debug:
        print("Starting the prefilling stage...", end="")
    
    eos_token_id = tokenizer.convert_tokens_to_ids('.')
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
    
    pre_fill_end_time = time.time()
    pre_fill_elapsed_time = pre_fill_end_time - pre_fill_start_time
    if show_debug:
        print(f"Done. Prefilling stage calculated in {pre_fill_elapsed_time:.2f} seconds.\n")

    generated = input_ids
    
    print(ori_prompt) 
    start_time = time.time()
    counter = 0
    for _ in range(num_tokens_to_generate):
        with torch.no_grad():
            outputs = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values
    
        next_token_logits = logits[:, -1, :]

        if sampling_method == "greedy":
            generated, next_token_id, next_token_text = greedy_sampling(next_token_logits, tokenizer, generated)
        elif sampling_method == "top-p":
            generated, next_token_id, next_token_text = top_p_sampling(next_token_logits, tokenizer, top_p, generated)
        
        print(next_token_text, end='', flush=True)
        if next_token_id.item() == eos_token_id:
            break
        if '.' in next_token_text:
            break
        counter += 1
    end_time = time.time()
    
    num_generated_tokens = generated.size(1) - input_ids.size(1)
    
    elapsed_time = end_time - start_time
    tokens_per_second = num_generated_tokens / elapsed_time
    
    if show_debug:
        print(f'\n\nGenerated {num_generated_tokens} tokens in {elapsed_time:.2f} seconds.')
        print(f'Decoding speed: {tokens_per_second:.2f} tokens/second')
    
    # if (method == 'dynamic_cut'):
    #     activation_ratios = []
    #     for layer in model.custom_layers:
    #         if (layer.activation_ratio > 0):
    #             activation_ratios.append(layer.activation_ratio)
    #     mean_activation_ratio = torch.Tensor(activation_ratios).mean()
    #     print("\nMean activation ratio: {}".format(mean_activation_ratio))





def main(method, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, prompt, memory_limit, num_fewshot, task_type, num_tokens_to_generate, device, sampling_method, siot_method_config, cluster_path = None, cpu_only = False, top_p = None, sparsity_levels_path = None, hybrid_split = None, model_neurons_filepath = None, function: str = "normal"):
    
    if (USE_SIOT_IMPROVEMENTS and method in ["siot", "dense"]):
        create_neurons_mask.main(start_num, end_num, siot_method_config)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(f"{script_dir}/transformers/siot_variables/mask_filepath.txt", "r") as f:
            MASK_FILEPATH = f.readlines()[0]
        print(f"SIOT: Use Mask for partial loading, mask file: {MASK_FILEPATH}") 
    
    model, tokenizer, num_layers = load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit)
    
    model = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, siot_method_config, cluster_path, cpu_only, sparsity_levels_path, hybrid_split, model_neurons_filepath)

    if (function == "normal"):
        generate(method, model, tokenizer, prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p)
    elif (function == "chat"):
        prompt = ""
        while True:
            prompt = input("User: ")
            if (prompt == "exit"):
                break
            generate(method, model, tokenizer, prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p, show_debug=False)
    elif (function == "predefined_prompts"):
        predefined_prompts = [
            "You visit a museum with ancient Greek artifacts. Which ancient Greek philosopher is known for his method of questioning and debate?",
            "You walk through a dense rainforest and hear monkeys chattering in the trees. What is the name of the large tropical forest found near the equator?",
            "You enjoy a cup of hot chocolate on a cold winter day. What ingredient is typically used to make hot chocolate?",
            "You read a novel about a detective solving mysteries. Who is the famous detective created by Sir Arthur Conan Doyle?",
            "You listen to a rock band perform a famous song on stage. Which band is known for the song \"Bohemian Rhapsody?\"",
            "You read about the Industrial Revolution. What were the key technological advancements during the Industrial Revolution?",
            "You read about the exploration of the New World by Christopher Columbus. What were the motivations for European exploration?",
            "You explore a museum with dinosaur skeletons on display. What is the name of the prehistoric period when dinosaurs lived?",
            "You’re at a concert listening to a famous pop artist. Who is known as the \"King of Pop?\"",
            "You take a selfie on top of the Empire State Building. In which U.S. city is the Empire State Building located?",
            "You’re planting a tree in your backyard. What is the process by which trees make their own food?",
            "What is the capital of Germany?",
        ]
        for prompt in predefined_prompts:
            generate(method, model, tokenizer, prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p, show_debug=False)
    else:
        raise ValueError("No function set: set --function")





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

    parser.add_argument('--function', type=str, choices=['normal', 'chat', 'predefined_prompts'], default='normal', help='Function to use')


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

    main(args.method, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity, args.prompt, args.memory_limit, args.num_fewshot, args.task_type, args.num_tokens_to_generate, args.device, args.sampling_method, siot_method_config, args.cluster_path, args.cpu_only, args.top_p, args.sparsity_levels_path, args.hybrid_split, args.model_neurons_filepath, args.function)
