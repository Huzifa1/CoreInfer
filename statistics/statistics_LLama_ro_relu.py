import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# === CONFIG ===
model_name = "meta-llama/Llama-3.2-3B"
max_new_tokens = 30
output_file = "non_relu.statistics"
dataset_file = "dataset.txt"
prompt_limit = 30000  # Limit how many prompts to process
do_sample = True  # Set to True if you want to sample instead of greedy generation
decoding_stage = False

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()
tokenCount = 0

# === Global neuron activation count dictionary ===
neuron_activation_sums = {}

# === Hook function to directly count neuron activations (no storing tensors) ===
def get_activation_hook(layer_id):
    def hook(module, input, output):
        x = input[0]
        batch, seq_len, hidden_dim = x.shape

        if layer_id not in neuron_activation_sums:
            neuron_activation_sums[layer_id] = torch.zeros(hidden_dim, dtype=torch.float32)
        
        neuron_activation_sums[layer_id] += x.view(-1, hidden_dim).abs().sum(dim=0).cpu()
    return hook

# === Register hooks ONCE globally ===
hooks = []
if decoding_stage:
    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_hook(get_activation_hook(f"layer_{i}_mlp_down_proj"))

# === Count how many prompts were already processed ===
processed_count = 0
try:
    with open(output_file, "r") as stats_file:
        for line in stats_file:
            if line.startswith("prompt: "):
                processed_count += 1
except FileNotFoundError:
    pass

# === Main function to process one prompt ===
def process_prompt(prompt):
    global tokenCount
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokenCountInput = inputs["input_ids"].shape[-1]
    if decoding_stage:
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=50,
                do_sample=do_sample
            )
            tokenCountOutput = generated.shape[-1] - tokenCountInput
            text_out = tokenizer.decode(generated[0], skip_special_tokens=True)
    else:
        with torch.no_grad():
            for i, layer in enumerate(model.model.layers):
                hook = layer.mlp.down_proj.register_forward_hook(get_activation_hook(f"layer_{i}_mlp_down_proj"))
                hooks.append(hook)
            tokenCountOutput = 0
            model(**inputs)
            for hook in hooks:
                hook.remove()
    tokenCount += tokenCountInput + tokenCountOutput
    return prompt

# === Process dataset file, prompt-by-prompt ===
i = processed_count
with open(dataset_file, "r") as f:
    for _ in range(processed_count):
        next(f)  # Skip already-processed lines

    for line in tqdm(f, desc="Processing prompts"):
        prompt = line.strip()
        _ = process_prompt(prompt)


        i += 1
        if i%100 == 0:
            print(prompt)
        if i >= (processed_count + prompt_limit):  # Limit how many prompts to process
            break

# === Save total neuron activation statistics at the end ===
with open(output_file, "a") as f:
    f.write(f"layers: {len(neuron_activation_sums)}\n")
    f.write(f"statistics size: [{list(neuron_activation_sums.values())[0].shape[0]}]\n")
    for layer_name, activation_counts in neuron_activation_sums.items():
        layer_nb = layer_name.split("layer_")[1].split("_")[0]
        count_str = ",".join(map(str, activation_counts.tolist()))
        f.write(f"{layer_nb}: {count_str}\n")
    f.write(f"Number of tokens: {tokenCount}")
