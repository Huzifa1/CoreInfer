import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json

def pre_process_prompt(prompt, task_type):
    if task_type == 'QA':
        final_prompt = 'Question: ' + prompt + '\nAnswer: '
        
    elif task_type == 'summarize':
        pre_prompt = 'Summarize the following document:\n'
        final_prompt = pre_prompt + prompt
        
    elif task_type == 'translate_de_en':
        final_prompt = 'German phrase: ' + prompt + '\nEnglish phrase: '
        
    elif task_type == 'translate_ro_en':
        final_prompt = 'Romanian phrase: ' + prompt + '\nEnglish phrase: '
        
    elif task_type == 'translate_fr_en':
        final_prompt = 'French phrase: ' + prompt + '\nEnglish phrase: '
        
    elif task_type == 'translate_en_fr':
        final_prompt = 'English phrase: ' + prompt + '\nFrench phrase: '

    else:
        raise RuntimeError("Task_type must be one of QA, summarize, translate_de_en or translate_ro_en")
    
    return final_prompt

def make_generate_kwargs(inputs, max_new_tokens, tokenizer, model):
    # Longest input in the batch
    input_len = inputs["input_ids"].shape[1]
    ctx_len = int(getattr(model.config, "max_position_embeddings", 2048))

    # Ensure total length <= context window
    max_new_safe = max(1, min(max_new_tokens, ctx_len - input_len))
    kwargs = dict(
        max_new_tokens=max_new_safe,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    return kwargs

# === Hook function to directly count neuron activations (no storing tensors) ===
def get_activation_hook(model_name, layer_id):
    def hook(module, input, output):
        x = input[0].detach()
        
        if x.is_cuda:
            torch.cuda.synchronize()
            
        x_cpu = x.cpu()
        
        if "llama" in model_name:
            batch, seq_len, hidden_dim = x_cpu.shape
        elif "opt" in model_name:
            batch, hidden_dim = x_cpu.shape
        else:
            raise ValueError("Not supported model")
             
        if layer_id not in neuron_activation_sums:
            neuron_activation_sums[layer_id] = torch.zeros(hidden_dim, dtype=torch.float32)
        
        neuron_activation_sums[layer_id] += x_cpu.view(-1, hidden_dim).abs().sum(dim=0)
    return hook

# === Main function to process one prompt ===
def process_prompt(prompt, task_type, model_name):
    prompt = pre_process_prompt(prompt, task_type)
    global tokenCount
    
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=tokenizer.model_max_length).to(model.device)
        tokenCountInput = inputs["input_ids"].shape[-1]

        # Optional early sanity check
        if (inputs["input_ids"] >= model.config.vocab_size).any():
            raise ValueError("Invalid token id (>= vocab size).")

        gen_kwargs = make_generate_kwargs(inputs, max_new_tokens, tokenizer, model)
        generated = model.generate(**inputs, **gen_kwargs)

        tokenCountOutput = generated.shape[-1] - inputs["input_ids"].shape[-1]
        text_out = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        tokenCount += tokenCountInput + tokenCountOutput
    return prompt

# === BATCHING CHANGE: Added new batch-processing function ===
def process_prompt_batch(prompts, task_type, model_name):
    global tokenCount

    processed_prompts = [pre_process_prompt(p, task_type) for p in prompts]
    inputs = tokenizer(
        processed_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_attention_mask=True,
    ).to(model.device)

    # Optional early sanity check
    if (inputs["input_ids"] >= model.config.vocab_size).any():
        raise ValueError("Invalid token id (>= vocab size).")

    with torch.no_grad():
        gen_kwargs = make_generate_kwargs(inputs, max_new_tokens, tokenizer, model)
        generated = model.generate(**inputs, **gen_kwargs)

        # safer per-sample token counting (handles early EOS)
        tokenCount += sum(
            g.size(0) + i.size(0)
            for g, i in zip(generated, inputs["input_ids"])
        )
    
    return processed_prompts

# === CONFIG ===
configs = [
    {
        "output_file": "triviaqa.statistics",
        "dataset_file": "triviaqa.txt",
        "task_type": "QA"
    },
    {
        "output_file": "squadv2.statistics",
        "dataset_file": "squadv2.txt",
        "task_type": "QA"
    },
    {
        "output_file": "mlqa.statistics",
        "dataset_file": "mlqa.txt",
        "task_type": "QA"
    },
    {
        "output_file": "piqa.statistics",
        "dataset_file": "piqa.txt",
        "task_type": "QA"
    },
    {
        "output_file": "wmt14-fr-en.statistics",
        "dataset_file": "wmt14-fr-en.txt",
        "task_type": "translate_fr_en"
    },
    {
        "output_file": "wmt14-en-fr.statistics",
        "dataset_file": "wmt14-en-fr.txt",
        "task_type": "translate_en_fr"
    },
    
    {
        "output_file": "wmt16-de-en.statistics",
        "dataset_file": "wmt16-de-en.txt",
        "task_type": "translate_de_en"
    },
    {
        "output_file": "wmt16-ro-en.statistics",
        "dataset_file": "wmt16-ro-en.txt",
        "task_type": "translate_ro_en"
    },
    {
        "output_file": "cnn_dailymail.statistics",
        "dataset_file": "cnn_dailymail.json",
        "task_type": "summarize"
    },
    {
        "output_file": "samsum.statistics",
        "dataset_file": "samsum.json",
        "task_type": "summarize"
    },
    {
        "output_file": "xsum.statistics",
        "dataset_file": "xsum.json",
        "task_type": "summarize"
    }
]

model_name = "../models/opt-6.7b"
max_new_tokens = 128
prompt_limit = 5000  # Limit how many prompts to process
    
for config in configs:
    output_file = f"statistics_files/{config['output_file']}"
    dataset_file = f"datasets_files/{config['dataset_file']}"
    task_type = config["task_type"]

    # === Load model and tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    ctx_len = int(getattr(model.config, "max_position_embeddings", 2048))
    tokenizer.model_max_length = ctx_len
    tokenCount = 0

    # === Global neuron activation count dictionary ===
    neuron_activation_sums = {}

    # === Register hooks ONCE globally ===
    hooks = []
    if "llama" in model_name:
        for i, layer in enumerate(model.model.layers):
            layer.mlp.down_proj.register_forward_hook(get_activation_hook(model_name, f"layer_{i}_mlp_down_proj"))
    elif "opt" in model_name:
        for i, layer in enumerate(model.model.decoder.layers):
            layer.fc2.register_forward_hook(get_activation_hook(model_name, f"layer_{i}_fc2"))
    else:
        raise ValueError("Not supported model")

    # === Process dataset file, prompt-by-prompt ===
    ii = 0
    with open(dataset_file, "r") as f:
        batch_size = 16  # BATCHING CHANGE: Set batch size
        batch = []
                
        if "json" in dataset_file:
            prompts = json.load(f)
            for p in tqdm(prompts, desc="Processing prompts"):
                batch.append(p)
                if len(batch) == batch_size:
                    _ = process_prompt_batch(batch, task_type, model_name)  # BATCHING CHANGE
                    batch = []
                    ii += batch_size
                    if ii >= (prompt_limit):  # Limit how many prompts to process
                        break
            if batch:
                _ = process_prompt_batch(batch, task_type, model_name)  # BATCHING CHANGE
        else:
            for line in tqdm(f, desc="Processing prompts"):
                prompt = line.strip()
                if prompt:
                    batch.append(prompt)
                if len(batch) == batch_size:
                    _ = process_prompt_batch(batch, task_type, model_name)  # BATCHING CHANGE
                    batch = []
                    ii += batch_size
                    if ii >= (prompt_limit):  # Limit how many prompts to process
                        break
            if batch:
                _ = process_prompt_batch(batch, task_type, model_name)  # BATCHING CHANGE

    # === Save total neuron activation statistics at the end ===
    with open(output_file, "a") as f:
        f.write(f"layers: {len(neuron_activation_sums)}\n")
        f.write(f"statistics size: [{list(neuron_activation_sums.values())[0].shape[0]}]\n")
        for layer_name, activation_counts in neuron_activation_sums.items():
            layer_nb = layer_name.split("layer_")[1].split("_")[0]
            count_str = ",".join(map(str, activation_counts.tolist()))
            f.write(f"{layer_nb}: {count_str}\n")
        f.write(f"Number of tokens: {tokenCount}")
