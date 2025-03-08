import torch
import time
from pathlib import Path
from transformers import OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets import load_from_disk
import re
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os
import pickle
from utils import process_data

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
def _load_model_data(checkpoint_path, dataset_path, device, memory_limit):
    if memory_limit == True:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map=device, torch_dtype=torch.float16)
        
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, device_map=device)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    dataset = load_from_disk(dataset_path)
    print("Model and Data has been loaded.")
    
    return model, tokenizer, dataset



def get_activation(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = input[0].detach().cpu()
    return hook

def register_opt_act_hooks(model, activation_dict):
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.ReLU):
            hooks.append(layer.register_forward_hook(get_activation(name, activation_dict)))
    return hooks


def register_llama_act_hooks(model, activation_dict):
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.SiLU):
            hooks.append(layer.register_forward_hook(get_activation(name, activation_dict)))
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

# Use elbow method to decide k
def Elbow(encoded_data):
    # An empty list to store the inertia (sum of squared distances of samples to their nearest cluster center)
    # for different numbers of clusters.
    # Inertia is a measure of how well the clusters fit the data: Lower inertia = tighter clusters.
    wcss = []
    for i in range(1, 11):
        # Loop from 1 to 10 clusters
        kmeans = KMeans(n_clusters=i, random_state=42)
        # Try to create `i` clusters based on the data
        kmeans.fit_predict(encoded_data)
        # Append the inertia to wcss
        wcss.append(kmeans.inertia_)
    
    # Example: wcss = [5000, 3200, 2500, 2100, 1900, 1800, 1700, 1650, 1600, 1580]
    # differences = [-1800, -700, -400, -200, -100, -100, -50, -50, -20]
    # second_differences = [1100, 300, 200, 100, 0, 50, 0, 30]
    differences = np.diff(wcss)
    second_differences = np.diff(differences)

    # Finds the index of the maximum second difference, which indicates the sharpest change in inertia reduction.
    # +2 is because len(wcss) = 10, len(differences) = 9, len(second_differences) = 8
    elbow_k = np.argmax(second_differences) + 2
    if elbow_k<4:
        # There should be a minimum of 4 clusters
        elbow_k=4
    print(f"Optimal number of clusters by Automated Elbow Method: {elbow_k}")
    
    # Basically, the elbow is a point where the inertia decrease slows down sharply
    # So it's the best point (best number of clusters)
    return elbow_k



def cluster_opt_data(model, tokenizer, precessed_data, Layer_num, device):
    data = precessed_data
    activations = []
    for i in range(len(data)):
        prompt = data[i]
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', prompt)
        tokenized_input = tokenizer(cleaned_text, return_tensors="pt", max_length=100, truncation=True).to(device)
    
        activation_dict = {}
        hooks2 = register_opt_act_hooks(model, activation_dict)
    
        with torch.no_grad():
            outputs = model(**tokenized_input)
            
        activations.append(activation_dict)
        remove_hooks(hooks2)
    
    layer_act=[]
    sentence_lenth=[]
    for i in range(len(activations)):
        str_act="model.decoder.layers."+str(Layer_num)+".activation_fn"
        tensor = activations[i][str_act].cpu()
        layer_act.append(tensor)
        m=tensor.size(0)
        sentence_lenth.append(m)
        
    act_all= torch.cat(layer_act, dim=0)
    
    sentence=[]
    num=0
    for i in range(len(sentence_lenth)):
        lenth=sentence_lenth[i]
        list_now=list(range(num, num+lenth))
        sentence.append(list_now)
        num=num+lenth
    
    count_act_all = (act_all > 0).sum(dim=1)
    sorted_values, sorted_indices = torch.sort(act_all, dim=1, descending=True)
    
    top50_indices=[]
    for i in range(act_all.size(0)):
        indices = sorted_indices[i, :int(torch.round(count_act_all[i]*0.4))]
        top50_indices.append(indices.tolist())
    
    SEN_F=[]
    for i in range(len(sentence)):
        cluster5= sentence[i]
        act_clu =  [top50_indices[i] for i in cluster5]
        data_flattened = [item for sublist in act_clu for item in sublist]
        data_flattened=torch.tensor(data_flattened)
        unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
    
        sorted_indices = torch.argsort(counts, descending=True)
        sorted_numbers = unique_numbers[sorted_indices]
        sorted_counts = counts[sorted_indices]
        SEN_F.append(sorted_numbers[:3000].numpy())

    return SEN_F




def cluster_llama_data(model, tokenizer, precessed_data, Layer_num, device):
    data = precessed_data
    activations = []
    for i in range(len(data)):
        prompt = data[i]
        tokenized_input = tokenizer(prompt, return_tensors="pt").to(device)
        
        activation_dict = {}
        hooks = register_llama_act_hooks(model, activation_dict)
    
        with torch.no_grad():
            outputs = model(**tokenized_input)
    
        activations.append(activation_dict)
    
        remove_hooks(hooks)
        
        del outputs
        del tokenized_input
        del activation_dict
    
    # The above code basically gets the activations for each layer of the model for each input in the dataset
    # They are sotred in `activations`

    # This will contain the activations of the `Layer_num` layer for each item in the dataset
    layer_act=[]
    # This will contain the token length of each item of the dataset
    sentence_lenth=[]
    for i in range(len(activations)):
        str_act="model.layers."+str(Layer_num)+".mlp.act_fn"
        tensor = activations[i][str_act].cpu()
        layer_act.append(tensor.squeeze(0))
        m=tensor.size(1)
        sentence_lenth.append(m)
    # This will be a tensor of the `layer_act` tensors
    A_tensor = torch.cat(layer_act, dim=0)

    # Each item of this will be a list of range from token_start to token_end
    # For example, assuming first 2 items of the dataset have 24 and 18 tokens
    # Then the first item will be [0, 1, ...., 23]
    # The second item will be [24, 25, ...., 41]
    sentence=[]
    num=0
    for i in range(len(sentence_lenth)):
        lenth=sentence_lenth[i]
        list_now=list(range(num, num+lenth))
        sentence.append(list_now)
        num=num+lenth
    
    # Assume act_all is of size [42, 14336]. (Because assume 2 sentences of 24 and 18 tokens)
    act_all=(A_tensor).cpu()
    # This will store for each token, the count of positive activations. It will be of size [42]
    count_act_all = (act_all > 0).sum(dim=1)
    # Contain the indices of the `act_all` sorted based on their activation values. It will be of same size as `act_all`
    sorted_values, sorted_indices = torch.sort(act_all, dim=1, descending=True)
    
    top50_indices=[]
    for i in range(act_all.size(0)):
        # For each token, select top 40% postive activations
        indices = sorted_indices[i, :int(torch.round(count_act_all[i]*0.4))]
        top50_indices.append(indices.tolist())

    SEN_F=[]
    for i in range(len(sentence)):
        # For each sentence, get the range of tokens
        # E.g, for first sentence, cluster5=range(0, 24).
        cluster5= sentence[i]
        # Get the top 40% positive activations for each token in the sentence
        act_clu =  [top50_indices[i] for i in cluster5]
        # Flatten the results into 1D array and transform it to a tensor
        data_flattened = [item for sublist in act_clu for item in sublist]
        data_flattened=torch.tensor(data_flattened)

        # From here and on is very similar to the code in the `forward`
        # function in the `DownLayer` class in the `convert_llama_model.py` file
        unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)

        sorted_indices = torch.argsort(counts, descending=True)
        sorted_numbers = unique_numbers[sorted_indices]
        sorted_counts = counts[sorted_indices]
        SEN_F.append(sorted_numbers[:3000].numpy())

    # Basically, each item of this list will be a list containing the core neurons of each sentence in the dataset
    return SEN_F



def save_cluster(model_name, SEN_F,cluster_path):
    data = SEN_F
    if "opt" in model_name:
        mlb = MultiLabelBinarizer(classes=range(0, 16383))
    elif "llama" in model_name:
        mlb = MultiLabelBinarizer(classes=range(0, 14336))
    encoded_data = mlb.fit_transform(data)

    # Basically, `encoded_data` will be matrix of size [len(data), 14336]
    # It's basically a binary matrix where each row represents a sample from the `data`,
    # and each column represents a neuron (total 14336 columns=neurons).
    # If the neuron index is available in `data`, then its relative index in the matrix
    # will be set to 1, otherwise 0


    k = Elbow(encoded_data)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(encoded_data)
    

    file_path = f'{cluster_path}/cluster_activation/cluster.pkl'
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(clusters, f)
        
    print(f"Clusters saved to {file_path}")
    
    file_path = f'{cluster_path}/cluster_activation/kmeans.pkl'
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(kmeans, f)

    with open(f'{cluster_path}/cluster_activation/mlb_model.pkl', 'wb') as file:
        pickle.dump(mlb, file)

    return clusters, k





def save_opt_neurons(model, tokenizer, sentence, cluster, cluster_path, sparsity, k, device):
    parent_directory = f'{cluster_path}/neuron_activation'

    os.makedirs(parent_directory, exist_ok=True)
    for i in range(32):
        folder_name = str(i)
        folder_path = os.path.join(parent_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)
    
    value_ratio = 0.4
    
    model.cuda()
    activations = []
    
    for cluster_num in range(0,k):
        positions = np.where(cluster == cluster_num)[0]
        data = [sentence[i] for i in positions]
        for i in range(len(data)):
            prompt = data[i]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            activation_dict = {}
            hooks2 = register_opt_act_hooks(model, activation_dict)
        
            with torch.no_grad():
                outputs = model(**inputs)
            activations.append(activation_dict)
            remove_hooks(hooks2)
        
        for Layer_num in range(0,32):
            layer_act=[]
            for i in range(len(activations)):
                str_act="model.decoder.layers."+str(Layer_num)+".activation_fn"
                tensor = activations[i][str_act].cpu()
                layer_act.append(tensor)
            A_tensor = torch.cat(layer_act, dim=0)
            sorted_values, sorted_indices = torch.sort(A_tensor, dim=1, descending=True)
            limit=int(value_ratio*(A_tensor>0).sum().item()/A_tensor.size(0))
            top_indices = sorted_indices[:, :limit]
            data_flattened = top_indices.reshape(-1)
            unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
            sorted_indices = torch.argsort(counts, descending=True)
            sorted_indices_clu = unique_numbers[sorted_indices]
            remained_num = int(len(sorted_indices_clu) * sparsity)
            indices_all=sorted_indices_clu[:remained_num]
            
            with open(f'{cluster_path}/neuron_activation/{Layer_num}/cluster_{cluster_num}.pkl', 'wb') as f:
                pickle.dump(indices_all, f)










def save_llama_neurons(model, tokenizer, sentence, cluster, cluster_path, sparsity, k, device):
    parent_directory = f'{cluster_path}/neuron_activation'

    os.makedirs(parent_directory, exist_ok=True)
    for i in range(32):
        folder_name = str(i)
        folder_path = os.path.join(parent_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)
    
    value_ratio = 0.4
    
    model.cuda()
    activations = []
    
    for cluster_num in range(0,k):
        # Loop over every cluster
        positions = np.where(cluster == cluster_num)[0]
        # Get the sentences that belong to this cluster
        data = [sentence[i] for i in positions]

        # Get the activations of each sentence for each layer and append it to activations
        for i in range(len(data)):
            prompt = data[i]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            activation_dict = {}
            hooks = register_llama_act_hooks(model, activation_dict)
        
            with torch.no_grad():
                outputs = model(**inputs)
            activations.append(activation_dict)
            remove_hooks(hooks)
        
        # Loop over each layer, get the core neurons and store them an a file
        # This code is very similar to the code in the `forward`
        # function in the `DownLayer` class in the `convert_llama_model.py` file
        for Layer_num in range(0,32):
            layer_act=[]
            for i in range(len(activations)):
                str_act="model.layers."+str(Layer_num)+".mlp.act_fn"
                tensor = activations[i][str_act].cpu()
                layer_act.append(tensor.squeeze(0))
            A_tensor = torch.cat(layer_act, dim=0)
            sorted_values, sorted_indices = torch.sort(A_tensor, dim=1, descending=True)
            limit=int(value_ratio*(A_tensor>0).sum().item()/A_tensor.size(0))
            top_indices = sorted_indices[:, :limit]
            data_flattened = top_indices.reshape(-1)
            unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
            sorted_indices = torch.argsort(counts, descending=True)
            sorted_indices_clu = unique_numbers[sorted_indices]
            remained_num = int(len(sorted_indices_clu) * sparsity)
            indices_all=sorted_indices_clu[:remained_num]
            
            with open(f'{cluster_path}/neuron_activation/{Layer_num}/cluster_{cluster_num}.pkl', 'wb') as f:
                pickle.dump(indices_all, f)





def main(model_name, dataset_name, checkpoint_path, dataset_path, memory_limit, Layer_num, cluster_path, sparsity, device):
    
    model, tokenizer, dataset = _load_model_data(checkpoint_path, dataset_path, device, memory_limit)
    
    precessed_data = process_data(dataset, dataset_name)
    
    if "opt" in model_name:
        SEN_F = cluster_opt_data(model, tokenizer, precessed_data, Layer_num, device)
    elif "llama" in model_name:
        SEN_F = cluster_llama_data(model, tokenizer, precessed_data[:5], Layer_num, device)
    
    cluster, k = save_cluster(model_name, SEN_F, cluster_path)

    if "opt" in model_name:
        save_opt_neurons(model, tokenizer, precessed_data, cluster, cluster_path, sparsity, k, device)
    elif "llama" in model_name:
        save_llama_neurons(model, tokenizer, precessed_data, cluster, cluster_path, sparsity, k, device)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="opt-6.7b", help='Model Name')
    parser.add_argument('--dataset_name', type=str, default="truthful_qa", help='Dataset Name')
    parser.add_argument('--Layer_num', type=int, default=25, help='Number of cluster layer.')
    parser.add_argument('--checkpoint_path', type=Path, help='Model checkpoint path.')
    parser.add_argument('--dataset_path', type=Path, help='Dataset path.')
    parser.add_argument('--cluster_path', type=Path, help='Save path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--sparsity', type=float, default=0.4, help='Sparsity level.')
    parser.add_argument('--memory_limit', action='store_true', help='Enable memory limit.')

    args = parser.parse_args()

    main(args.model_name, args.dataset_name, args.checkpoint_path, args.dataset_path, args.memory_limit, args.Layer_num, args.cluster_path, args.sparsity, args.device)
