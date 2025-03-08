import torch.nn as nn
import gc
from scipy.stats import norm
import torch
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import pickle
import os
global_cluster = None


class DownLayer(nn.Module):
    def __init__(self, weight, act_list, num, kmeans, mlb_loaded, sparsity, memory_limit,name = None):
        super(DownLayer, self).__init__()
        self.weight = weight.clone()
        remained_neurons = int(weight.size(1) * sparsity)
        self.remained_neurons = remained_neurons
        self.memory_limit = memory_limit
        if memory_limit:
            self.filtered_W = torch.zeros((weight.size(0),remained_neurons)).to(torch.float16).cpu()
        else:
            self.filtered_W = torch.zeros((weight.size(0),remained_neurons)).to(torch.float16).cuda()
        self.act_list = act_list[:][:remained_neurons]

        self.num = num
        self.kmeans = kmeans
        self.mlb_loaded=mlb_loaded
        self.weight_updated = False
        
    def forward(self, x):
        if x.size(1)>1:
            # If in the prefilling stage

            self.weight_updated = False
            self.weight = self.weight
            true_value = x @ self.weight.T
            
            x1=x.clone()
            if self.num == 25:
                # Take activations from layer 25 specifically

                # This code is very similar to the "DownLayer" in the stable method
                # Basically indices_all contains the indecies of the core neurons of the layer 25
                sorted_values, sorted_indices = torch.sort(x1, dim=1, descending=True)
                limit=int(0.4*(x1>0).sum().item()/x1.size(0))
                top_indices = sorted_indices[:, :limit]
                data_flattened = top_indices.reshape(-1)
                unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
                sorted_indices = torch.argsort(counts, descending=True)
                sorted_indices_clu = unique_numbers[sorted_indices]
                indices_all = sorted_indices_clu[:3000].cpu().numpy()

                # Using the kmeans, and the core neurons of the current input,
                # specifiy to which cluster does the current input belong to
                # Save the cluster number in `global_cluster`
                indices_all_2d = [indices_all.tolist()]
                new_data = self.mlb_loaded.transform(indices_all_2d)
                
                predictions = self.kmeans.predict(new_data)
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                global global_cluster
                global_cluster = predictions[0]

                del indices_all_2d, new_data
                gc.collect()
            
        
            
        else:
            # If in the decoding stage for the first time
            if not self.weight_updated:
                cluster_num = global_cluster
                # Get the indecies of the core neurons of the cluster
                activated_list = self.act_list[cluster_num].tolist()
                # Set the filtered weights to the core neurons only
                self.filtered_W = self.weight[:, activated_list].clone()
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            # Calculate using the core neurons only
            true_value = x @ self.filtered_W.T
            
        return true_value



class UpLayer(nn.Module):
    def __init__(self, weight, act_list, sparsity, memory_limit, name = None):
        super(UpLayer, self).__init__()
        self.weight = weight.clone()
        remained_neurons = int(weight.size(0) * sparsity)
        self.remained_neurons = remained_neurons
        self.memory_limit = memory_limit
        if memory_limit:
            self.filtered_W = torch.zeros((remained_neurons,weight.size(1))).to(torch.float16).cpu()
        else:
            self.filtered_W = torch.zeros((remained_neurons,weight.size(1))).to(torch.float16).cuda()
        self.act_list = act_list[:][:remained_neurons]
        self.weight_updated = False
        
    def forward(self, x):
        if x.size(1)>1:
            # If in the prefilling stage, calculate normally using all weights
            self.weight_updated = False
            self.weight = self.weight
            true_value = x @ self.weight.T
            if self.memory_limit:
                self.weight = self.weight.cpu()

        else:
            if not self.weight_updated:
                # If in the decoding stage for the first time,
                # Get the activations of the global cluster
                # Set the filtered weights to the core neurons only
                cluster_num = global_cluster
                activated_list = self.act_list[cluster_num].tolist()
                self.filtered_W = self.weight[activated_list,:].clone()
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            # Calculate using the core neurons only
            true_value = x @ self.filtered_W.T
            
        return true_value


class GateLayer(nn.Module):
    def __init__(self, weight, act_list, sparsity, memory_limit, name = None):
        super(GateLayer, self).__init__()
        self.weight = weight.clone()
        remained_neurons = int(weight.size(0) * sparsity)
        self.remained_neurons = remained_neurons
        self.memory_limit = memory_limit
        if memory_limit:
            self.filtered_W = torch.zeros((remained_neurons, weight.size(1))).to(torch.float16).cpu()
        else:
            self.filtered_W = torch.zeros((remained_neurons, weight.size(1))).to(torch.float16).cuda()
        self.act_list = act_list[:][:remained_neurons]
        self.weight_updated = False
        
    def forward(self, x):
        if x.size(1)>1:
            self.weight_updated = False
            self.weight = self.weight
            true_value = x @ self.weight.T
            if self.memory_limit:
                self.weight = self.weight.cpu()

        else:
            if not self.weight_updated:
                cluster_num = global_cluster
                activated_list = self.act_list[cluster_num].tolist()
                self.filtered_W = self.weight[activated_list,:].clone()
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            
            true_value = x @ self.filtered_W.T
            
        return true_value





def convert_llama_model_sim(model, num_layers, sparsity, start_num, end_num, memory_limit, cluster_path):
    
    with open(f'{cluster_path}/cluster_activation/kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    with open(f'{cluster_path}/cluster_activation/mlb_model.pkl', 'rb') as file:
        mlb_loaded = pickle.load(file)
        
    
    cluster_num = 0
    for root, dirs, files in os.walk(f'{cluster_path}/neuron_activation/0'):
        cluster_num += len(files)

    # This array will be of length (num_layers)
    # Each item will be another list of length (cluster_num)
    # Each item in this second list will contain the core neurons of a cluster
    Predictor=[] 
    for i in range(num_layers):
        predict_layer=[]
        for cluster_catagory in range(0,cluster_num):
            with open(f'{cluster_path}/neuron_activation/{i}/cluster_{cluster_catagory}.pkl', 'rb') as f:
                    model_predictor = pickle.load(f)
            predict_layer.append(model_predictor)
        Predictor.append(predict_layer)
    
    
    for name, module in tqdm(model.named_modules(), desc="Convert Llama Models"):
        if "down" in name or "up" in name or "gate" in name:
            num=int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                if "down" in name:    
                    NewLayer = DownLayer(module.weight, Predictor[num], num, kmeans, mlb_loaded, sparsity, memory_limit, name = name)
                elif "up" in name:
                    NewLayer = UpLayer(module.weight, Predictor[num], sparsity, memory_limit, name = name)
                elif "gate" in name:
                    NewLayer = GateLayer(module.weight, Predictor[num], sparsity, memory_limit, name = name)
                setattr(parent, attr_name, NewLayer)
                del module
    
    gc.collect()
    print("Converted Model Done")
    
    return model