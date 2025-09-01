import time
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import compute_module_sizes

# === Settings ===
model_name = "../models/llama3-3b"   # Replace with any supported model
offload_dir = "./offload"         # Must be a fast disk path (e.g. NVMe)
prompt = "The future of AI is"    # Your input prompt
max_new_tokens = 50               # Number of tokens to generate

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Load model config and initialize on meta device ===
config = AutoConfig.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# === Tie weights before loading checkpoint ===
model.tie_weights()

# === Force all layers to be offloaded to disk ====
device_map = {name: "cpu" for name in compute_module_sizes(model)}

# === Load checkpoint with forced disk offloading ===
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=model_name,
    device_map=device_map,
    offload_folder=offload_dir,
    offload_buffers=True,
    dtype=torch.float16  # Use float32 if float16 unsupported
)

# === Tokenize the prompt and move to correct device ===
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# === Run inference and time it ===
start_time = time.time()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
end_time = time.time()

# === Decode output ===
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# === Measure performance ===
elapsed_time = end_time - start_time
num_generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
tokens_per_second = num_generated_tokens / elapsed_time

# === Print results ===
print("\nGenerated text:\n")
print(generated_text)
print("\n--- Performance ---")
print(f"Time taken: {elapsed_time:.2f} seconds")
print(f"Tokens generated: {num_generated_tokens}")
print(f"Tokens/sec: {tokens_per_second:.2f}")
