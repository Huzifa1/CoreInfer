import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common

global_cluster = None


class ReduceLayer(nn.Module):
    def __init__(self, weight, bias, act_list, num, kmeans, mlb_loaded, sparsity, memory_limit, cpu_only):
        super(ReduceLayer, self).__init__()
        
        remained_neurons = int(weight.size(0) * sparsity)

        if bias is not None:
            self.bias = bias.clone()
        self.weight = weight.clone()
        self.remained_neurons = remained_neurons
        self.memory_limit = memory_limit
        self.cpu_only = cpu_only
        self.filtered_W = torch.zeros((remained_neurons,weight.size(1))).to(torch.float16).cpu()
        self.filtered_bias = torch.zeros(remained_neurons).to(torch.float16).cpu()
        self.act_list = [sublist[:remained_neurons] for sublist in act_list]
        self.num = num
        self.kmeans = kmeans
        self.mlb_loaded = mlb_loaded
        self.weight_updated = False
        
    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")

        if x.size(0)>1:
            self.weight_updated = False
            self.weight = self.weight.to(device)
            self.bias = self.bias.to(device)
            true_value = x @ self.weight.T + self.bias

            if self.num == 25:
                # Here, for some reason they pick top 3000 neurons from the sorted_indices_clu
                # So to match the function params, set sparsity to 1 and neuron_num to 3000
                indices_all = common.get_core_neurons(true_value, 0.4, 1, 3000)

                indices_all_2d = [indices_all.tolist()]
                new_data = self.mlb_loaded.transform(indices_all_2d)
                
                predictions = self.kmeans.predict(new_data)
                if self.memory_limit or self.cpu_only:
                    self.weight = self.weight.to("cpu")
                    self.bias = self.bias.to("cpu")
                    
                global global_cluster
                global_cluster = predictions[0]
                del indices_all_2d, new_data
                gc.collect()
        
            
        else:
            if not self.weight_updated:
                cluster_num = global_cluster
                activated_list = self.act_list[cluster_num].tolist()
                self.filtered_W = self.weight[activated_list,:].clone().to(device)
                self.filtered_bias = self.bias[activated_list].clone().to(device)
                if self.memory_limit or self.cpu_only:
                    self.weight = self.weight.cpu()
                    self.bias = self.bias.cpu()
                self.weight_updated = True

            true_value = x @ self.filtered_W.T + self.filtered_bias
            
        return true_value



class ReduceLayer_fc2(nn.Module):
    def __init__(self, weight, bias, act_list, sparsity, memory_limit, cpu_only):
        super(ReduceLayer_fc2, self).__init__()

        remained_neurons = int(weight.size(1) * sparsity)

        if bias is not None:
            self.bias = bias.clone()
        self.weight = weight.clone()
        self.remained_neurons = remained_neurons
        self.memory_limit = memory_limit
        self.cpu_only = cpu_only
        self.filtered_W = torch.zeros((weight.size(0),remained_neurons)).to(torch.float16).cpu()
        self.act_list = [sublist[:remained_neurons] for sublist in act_list]
        self.weight_updated = False
        
    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")

        if x.size(0)>1:
            self.weight = self.weight.to(device)
            self.bias = self.bias.to(device)
            true_value = x @ self.weight.T + self.bias

        else:
            if not self.weight_updated:
                cluster_num = global_cluster
                activated_list = self.act_list[cluster_num].tolist()
                self.filtered_W = self.weight[:,activated_list].clone().to(device)
                if self.memory_limit or self.cpu_only:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            
            true_value = x @ self.filtered_W.T.to(device) + self.bias.to(device)
            
        return true_value




def convert_opt_model_sim(model, num_layers, sparsity, start_num, end_num, memory_limit, cluster_path, cpu_only):
    
    kmeans, mlb_loaded, Predictor = common.read_cluster_files(cluster_path, num_layers)

    for name, module in tqdm(model.named_modules(), desc="Convert Opt Models"):
        if "fc1" in name or "fc2" in name:
            num=int(name.split('.')[3])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    
                if "fc1" in name:
                    NewLayer = ReduceLayer(module.weight, module.bias, Predictor[num], num, kmeans, mlb_loaded, sparsity, memory_limit, cpu_only)
                else:
                    NewLayer = ReduceLayer_fc2(module.weight, module.bias, Predictor[num], sparsity, memory_limit, cpu_only)
                setattr(parent, attr_name, NewLayer)
                del module
    
    gc.collect()
    
    return model