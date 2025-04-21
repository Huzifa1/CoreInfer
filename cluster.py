import torch
from pathlib import Path
from datasets import load_from_disk
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os
import pickle
from utils import process_data
from common import *

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_cluster(model_name, SEN_F, cluster_path):
    data = SEN_F

    num_neurons = MODEL_INFO[model_name]['num_neurons']
    mlb = MultiLabelBinarizer(classes=range(0, num_neurons))

    encoded_data = mlb.fit_transform(data)
    
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



def save_neurons(activations, model_name, model, cluster, cluster_path, sparsity, k, device):
    parent_directory = f'{cluster_path}/neuron_activation'

    os.makedirs(parent_directory, exist_ok=True)
    for i in range(32):
        folder_name = str(i)
        folder_path = os.path.join(parent_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)
    
    value_ratio = 0.2
    num_neurons = MODEL_INFO[model_name]['num_neurons']

    model.to(device)
    
    for cluster_num in range(0, k):
        current_activations = []
        positions = np.where(cluster == cluster_num)[0]
        for i in positions:
            current_activations.append(activations[i])

        for Layer_num in tqdm(range(0, 32), desc=f"Saving neurons for cluster: {cluster_num}"):
           
            layer_act = []
            str_act = get_layer_name(model_name, Layer_num)
            for i in range(len(current_activations)):
                tensor = current_activations[i][str_act].cpu()
                if "llama" in model_name:
                    tensor = tensor.squeeze(0)
                layer_act.append(tensor)

            A_tensor = torch.cat(layer_act, dim=0)

            indices_all = get_core_neurons(A_tensor, value_ratio, sparsity, num_neurons)
            
            with open(f'{cluster_path}/neuron_activation/{Layer_num}/cluster_{cluster_num}.pkl', 'wb') as f:
                pickle.dump(indices_all, f)





def main(model_name, dataset_name, checkpoint_path, dataset_path, memory_limit, Layer_num, cluster_path, sparsity, device):
    
    model, tokenizer, num_layers = load_model(checkpoint_path, 5, 27, checkpoint_path, device, memory_limit)
    
    dataset = load_from_disk(dataset_path)
    
    precessed_data = process_data(dataset, dataset_name)
    
    activations = collect_activations(model_name, precessed_data, tokenizer, device, model)

    SEN_F = get_sentence_core_neurons(model_name, Layer_num, activations, 0.4, 1, 3000)
    
    cluster, k = save_cluster(model_name, SEN_F, cluster_path)
    
    save_neurons(activations, model_name, model, cluster, cluster_path, sparsity, k, device)




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
    parser.add_argument('--cpu_only', action='store_true', help='Run inference on CPU only.')

    args = parser.parse_args()

    if (args.cpu_only):
        default_device = 'cpu'
        args.device = 'cpu'

    if args.cpu_only and args.memory_limit:
        parser.error("The options --cpu_only and --memory_limit cannot be used together.")

    main(args.model_name, args.dataset_name, args.checkpoint_path, args.dataset_path, args.memory_limit, args.Layer_num, args.cluster_path, args.sparsity, args.device)