import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common

global_cluster = None

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, act_list, num, kmeans, mlb_loaded, sparsity, memory_limit, cpu_only, name):
        super(CustomMLPLayer, self).__init__()

        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")
        
        if ("down" in name):
            remained_neurons = int(weight.size(1) * sparsity)
            self.filtered_W = torch.zeros((weight.size(0), remained_neurons)).to(torch.float16).to(device)
        else:
            remained_neurons = int(weight.size(0) * sparsity)
            self.filtered_W = torch.zeros((remained_neurons, weight.size(1))).to(torch.float16).to(device)

        
        self.name = name
        self.num = num
        self.kmeans = kmeans
        self.mlb_loaded = mlb_loaded
        self.weight = weight.clone()
        self.remained_neurons = remained_neurons
        self.memory_limit = memory_limit
        self.cpu_only = cpu_only
        self.act_list = [sublist[:remained_neurons] for sublist in act_list]
        self.weight_updated = False

    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")

        if x.size(1)>1:
            self.weight_updated = False
            self.weight = self.weight.to(device)
            true_value = x @ self.weight.T.to(device)
            if self.memory_limit or self.cpu_only:
                self.weight = self.weight.cpu()

            if "down" in self.name and self.num == 25:
                # Here, for some reason they pick top 3000 neurons from the sorted_indices_clu
                # So to match the function params, set sparsity to 1 and neuron_num to 3000
                indices_all = common.get_core_neurons(x.clone().squeeze(0), 0.4, 1, 3000)

                indices_all_2d = [indices_all.tolist()]
                new_data = self.mlb_loaded.transform(indices_all_2d)
                
                predictions = self.kmeans.predict(new_data)
                if self.memory_limit or self.cpu_only:
                    self.weight = self.weight.cpu()

                global global_cluster
                global_cluster = predictions[0]
                print("Current Input is predicted to be in cluster: ", global_cluster)

                del indices_all_2d, new_data
                gc.collect()

        else:
            if not self.weight_updated:
                cluster_num = global_cluster
                activated_list = self.act_list[cluster_num].tolist()
                if ("down" in self.name):
                    self.filtered_W = self.weight[:, activated_list].clone().to(device)
                else:
                    self.filtered_W = self.weight[activated_list,:].clone().to(device)
                if self.memory_limit or self.cpu_only:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            
            true_value = x @ self.filtered_W.T.to(device)
            
        return true_value


def convert_llama_model_sim(model, num_layers, sparsity, start_num, end_num, memory_limit, cluster_path, cpu_only):
    
    kmeans, mlb_loaded, Predictor = common.read_cluster_files(cluster_path, num_layers)
    
    for name, module in tqdm(model.named_modules(), desc = "Convert Llama Models"):
        if "down" in name or "up" in name or "gate" in name:
            num=int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
                
                NewLayer = CustomMLPLayer(module.weight, Predictor[num], num, kmeans, mlb_loaded, sparsity, memory_limit, cpu_only, name)
                setattr(parent, attr_name, NewLayer)
                del module
                
    gc.collect()
    
    return model