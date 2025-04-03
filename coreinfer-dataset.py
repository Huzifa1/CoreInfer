import torch
import time
from pathlib import Path
from utils import *
import sys
from common import *
from torch.nn.functional import softmax
from datasets import load_from_disk

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
def generate(method, model, tokenizer, ori_prompt, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p):
    model.eval()
    if method == 'stable_guided':
        prompt = process_prompt_stable(ori_prompt, task_type, num_fewshot)
    elif method == 'similarity_guided':
        prompt = process_prompt_similarity(ori_prompt, task_type)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    pre_fill_start_time = time.time()
    # print("Starting the prefilling stage...", end="")
    
    eos_token_id = tokenizer.convert_tokens_to_ids('.')
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
    
    pre_fill_end_time = time.time()
    pre_fill_elapsed_time = pre_fill_end_time - pre_fill_start_time
    # print(f"Done. Prefilling stage calculated in {pre_fill_elapsed_time:.2f} seconds.\n")

    generated = input_ids
    
    # print(ori_prompt) 
    start_time = time.time()
    
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
    
    end_time = time.time()
    
    num_generated_tokens = generated.size(1) - input_ids.size(1)
    
    elapsed_time = end_time - start_time
    tokens_per_second = num_generated_tokens / elapsed_time
    
    # print(f'\n\nGenerated {num_generated_tokens} tokens in {elapsed_time:.2f} seconds.')
    # print(f'Decoding speed: {tokens_per_second:.2f} tokens/second')





def main(method, model_name, checkpoint_path, sparsity, start_num, end_num, token_sparsity, max_items, memory_limit, num_fewshot, task_type, num_tokens_to_generate, device, dataset_name, sampling_method, cluster_path = None, cpu_only = False, top_p = None):
    model, tokenizer, num_layers = load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit)

    model = convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, cluster_path, cpu_only)

    print(f"Command: {' '.join(sys.argv)}\n\n")
    dataset = load_from_disk(f"./dataset/{dataset_name}")
    if max_items is None:
        max_items = len(dataset)
    for i in tqdm(range(0, max_items), desc="Generating Output"):
        if dataset_name == "wmt16-de-en":
            german_phrase = dataset["validation"][i]["translation"]["de"]
            english_phrase = dataset["validation"][i]["translation"]["en"]

            print("German Phrase: ", german_phrase)
            print("English Phrase: ", english_phrase)
            print("Model Response: ", end="")
            generate(method, model, tokenizer, german_phrase, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p)
            print("\n\n")

        elif dataset_name == "truthful_qa":
            question = dataset["validation"][i]["question"]
            best_answer = dataset["validation"][i]["best_answer"]

            print("Question: ", question)
            print("Best Answer: ", best_answer)
            print("Model Response: ", end="")
            generate(method, model, tokenizer, question, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p)
            print("\n\n")

        elif dataset_name == "trivia_qa":
            question = dataset["validation"][i]["question"]
            answer = dataset["validation"][i]["answer"]["value"]

            print("Question: ", question)
            print("Answer: ", answer)
            print("Model Response: ", end="")
            generate(method, model, tokenizer, question, task_type, num_fewshot, num_tokens_to_generate, device, sampling_method, top_p)
            print("\n\n")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="opt-6.7b", help='Model Name')
    parser.add_argument('--num_fewshot', type=int, default=6, help='Number of samples.')
    parser.add_argument('--num_tokens_to_generate', type=int, default=256, help='Maximum number of new tokens.')
    parser.add_argument('--task_type', type=str, choices=['QA', 'Summarize', 'translate_de_en'], default='QA', help='Type of task to perform.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--dataset_name', type=str, default="truthful_qa", choices=['truthful_qa', 'trivia_qa', 'wmt16-de-en'], help='Name of the dataset to use.')
    parser.add_argument('--sparsity', type=float, default=0.4, help='Sentence Sparsity level.')
    parser.add_argument('--start_num', type=int, default=5, help='Start layer.')
    parser.add_argument('--end_num', type=int, default=27, help='End layer.')
    parser.add_argument('--max_items', type=int, default=None, help='Maxixmum number of items in the dataset to run.')
    parser.add_argument('--top_p', type=float, default=0.9, help='When set, will use top-p sampling')
    parser.add_argument('--sampling-method', type=str, default="greedy", choices=["greedy", "top-p"], help='Choose sampling method')
    parser.add_argument('--token_sparsity', type=float, default=0.2, help='Token Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')
    parser.add_argument('--method', type=str, choices=['stable_guided', 'similarity_guided'], default='stable_guided', help='Method to use (default: stable_guided).')
    parser.add_argument('--cluster_path', type=str, default=None, help='Optional cluster path.')
    parser.add_argument('--cpu_only', action='store_true', help='Run inference on CPU only.')

    args = parser.parse_args()

    if (args.cpu_only):
        default_device = 'cpu'
        args.device = 'cpu'

    if args.cpu_only and args.memory_limit:
        parser.error("The options --cpu_only and --memory_limit cannot be used together.")

    main(args.method, args.model_name, args.checkpoint_path, args.sparsity, args.start_num, args.end_num, args.token_sparsity, args.max_items, args.memory_limit,
        args.num_fewshot, args.task_type, args.num_tokens_to_generate, args.device, args.dataset_name, args.sampling_method, args.cluster_path, args.cpu_only, args.top_p)