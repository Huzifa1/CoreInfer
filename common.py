from transformers import AutoModelForCausalLM, AutoTokenizer
from convert.convert_opt_model import convert_opt_model
from convert.convert_opt_model_sim import convert_opt_model_sim
from convert.convert_llama_model import convert_llama_model
from convert.convert_llama_model_sim import convert_llama_model_sim

from convert.convert_llama_model_dynamic_cut import convert_llama_model_dynamic_cut
from convert.convert_llama_model_static_cut import convert_llama_model_static_cut
from convert.convert_llama_model_dense import convert_llama_model_dense
from convert.convert_llama_model_moving_cut import convert_llama_model_moving_cut

from utils import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import re
import os
import pickle
from tqdm import tqdm
import time

MODEL_INFO = {
    'opt': {
        'num_neurons': 16384,
        'activation_fn': torch.nn.ReLU,
    },
    'llama': {
        'num_neurons': 14336,
        'activation_fn': torch.nn.SiLU
    }
}

def get_model_family(model_name):
    if "opt" in model_name:
        return "opt"
    elif "llama" in model_name:
        return "llama"
    else:
        # Currently only supports Opt and LLAMA models
        raise ValueError("Model Name not supported")
    
def get_layer_name(model_name, Layer_num):
    if "opt" in model_name:
        return f"model.decoder.layers.{Layer_num}.activation_fn"
    
    if "llama" in model_name:
        return f"model.layers.{Layer_num}.mlp.act_fn"
    
    raise ValueError("Model Name not supported")

def get_layer_number(model_name, layer_name):
    if "opt" in model_name:
        return int(layer_name.split(".")[3])
    
    if "llama" in model_name:
        return int(layer_name.split(".")[2])
    
    raise ValueError("Model Name not supported")


def load_model(model_name, start_num, end_num, checkpoint_path, device, memory_limit):
    start_time = time.time()

    if memory_limit == True:
        model, num_layers = load_model_memory_limit(checkpoint_path, start_num, end_num, model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map=device, torch_dtype=torch.float16)
        num_layers = model.config.num_hidden_layers
        
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, device_map=device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done. Loaded model in {elapsed_time:.2f} seconds.\n")
    return model, tokenizer, num_layers



def convert_model(method, model, model_name, num_layers, sparsity, start_num, end_num, token_sparsity, memory_limit, cluster_path=None, cpu_only = False):
    start_time = time.time()

    if "opt" in model_name:
        if method == 'stable_guided':
            model = convert_opt_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only)
        elif method == 'similarity_guided':
            model = convert_opt_model_sim(model, num_layers, sparsity, start_num, end_num, memory_limit, cluster_path, cpu_only)
            
        
    elif "llama" in model_name:
        if method == 'stable_guided':
            model = convert_llama_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only)
        elif method == 'similarity_guided':
            model = convert_llama_model_sim(model, num_layers, sparsity, start_num, end_num, memory_limit, cluster_path, cpu_only)
        elif method == 'dynamic_cut':
            model = convert_llama_model_dynamic_cut(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only)
        elif method == 'static_cut':
            model = convert_llama_model_static_cut(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only)
        elif method == 'dense':
            model = convert_llama_model_dense(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only)
        elif method == 'moving_cut':
            model = convert_llama_model_moving_cut(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only)
            
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done. Converted model in {elapsed_time:.2f} seconds.\n")
    return model



def get_core_neurons(x, token_sparsity, sparsity, neuron_num = None):
    sorted_values, sorted_indices = torch.sort(x, dim=1, descending=True)
    limit=int(token_sparsity * (x > 0).sum().item() / x.size(0))
    top_indices = sorted_indices[:, :limit]
    data_flattened = top_indices.reshape(-1)
    unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_indices_clu = unique_numbers[sorted_indices]

    if (neuron_num is not None):
        remained_neurons = int(neuron_num * sparsity)
    else: 
        # Then calculate them dynamically based on the size of `sorted_indices_clu`
        remained_neurons = int(len(sorted_indices_clu) * sparsity)

    indices_all = sorted_indices_clu[:remained_neurons].cpu()

    return indices_all

def get_core_neurons_improved(x, token_sparsity, sparsity, neuron_num):
    remained_neurons = int(neuron_num * sparsity)
    
    tokens_core_neurons = []
    for token_activation in x:
        _, sorted_indices = torch.sort(token_activation, descending=True)
        limit = int(token_sparsity * (token_activation > 0).sum().item())
        core_neurons = sorted_indices[:limit]
        tokens_core_neurons.append(core_neurons)
        
    data_flattened = [item for sublist in tokens_core_neurons for item in sublist]
    data_flattened = torch.tensor(data_flattened)
    unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)

    
    
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_indices_clu = unique_numbers[sorted_indices]
    indices_all = sorted_indices_clu[:remained_neurons].cpu()

    return indices_all


def get_activation(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = input[0].detach().cpu()
    return hook



def register_act_hooks(model_name, model, activation_dict, Layer_num = None):
    hooks = []

    model_family = get_model_family(model_name)
    model_instance = MODEL_INFO[model_family]['activation_fn']
    
    for name, layer in model.named_modules():
        if isinstance(layer, model_instance):
            if (Layer_num is None or Layer_num == get_layer_number(model_name, name)):
                hooks.append(layer.register_forward_hook(get_activation(name, activation_dict)))
            
    return hooks



def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()



def Elbow(encoded_data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit_predict(encoded_data)
        wcss.append(kmeans.inertia_)
    
    differences = np.diff(wcss)
    second_differences = np.diff(differences)

    elbow_k = np.argmax(second_differences) + 2
    if elbow_k < 4:
        print(f"Original K is {elbow_k}. It will be set to 4")
        elbow_k = 4
    print(f"Optimal number of clusters by Automated Elbow Method: {elbow_k}")
    
    return elbow_k



def collect_activations(model_name, data, tokenizer, device, model, Layer_num = None):
    activations = []
    for i in tqdm(range(len(data)), desc="Collecting activations"):

        prompt = data[i]
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', prompt)
        tokenized_input = tokenizer(cleaned_text, return_tensors="pt").to(device)
        
        activation_dict = {}
        hooks = register_act_hooks(model_name, model, activation_dict, Layer_num)
    
        with torch.no_grad():
            outputs = model(**tokenized_input)
    
        activations.append(activation_dict)
    
        remove_hooks(hooks)
        
        del outputs
        del tokenized_input
        del activation_dict
        
    return activations


def get_sentence_core_neurons(model_name, Layer_num, activations, token_sparsity, sparsity, neuron_num = None):
    
    SEN_F = []
    str_act = get_layer_name(model_name, Layer_num)
    for i in tqdm(range(len(activations)), desc="Calculating Core Neurons"):
        tensor = activations[i][str_act].cpu()

        if "llama" in model_name:
            tensor = tensor.squeeze(0)

        core_neurons = get_core_neurons(tensor, token_sparsity, sparsity, neuron_num)
        SEN_F.append(core_neurons.numpy())

    
    return SEN_F



def read_cluster_files(cluster_path, num_layers):
    with open(f'{cluster_path}/cluster_activation/kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    with open(f'{cluster_path}/cluster_activation/mlb_model.pkl', 'rb') as file:
        mlb_loaded = pickle.load(file)
        
    
    cluster_num = 0
    for root, dirs, files in os.walk(f'{cluster_path}/neuron_activation/0'):
        cluster_num += len(files)
        
    Predictor = []
    for i in range(num_layers):
        predict_layer = []
        for cluster_catagory in range(0, cluster_num):
            with open(f'{cluster_path}/neuron_activation/{i}/cluster_{cluster_catagory}.pkl', 'rb') as f:
                    model_predictor = pickle.load(f)
            predict_layer.append(model_predictor)
        Predictor.append(predict_layer)

    return kmeans, mlb_loaded, Predictor