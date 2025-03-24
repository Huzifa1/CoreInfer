# Convert Llama2 Model
import torch.nn as nn
from scipy.stats import norm
import gc
import torch
from tqdm import tqdm
import common

indices_list_all = []

class DownLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only):
        super(DownLayer, self).__init__()

        neuron_num = int(weight.size(1)*sparsity)
        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")

        self.weight = weight.clone()
        self.num = num
        self.token_sparsity = token_sparsity
        self.sparsity = sparsity
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit

        self.filtered_W = torch.zeros((weight.size(0),neuron_num)).to(torch.float16).to(device)

    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")

        if x.size(1)>1:
            self.weight_updated = False
            x_train=x.clone()
            
            true_value = x_train @ self.weight.T.to(device)

            squeezed_x = x.squeeze().clone()
            indices_all = common.get_core_neurons(squeezed_x, self.token_sparsity, self.sparsity, self.weight.size(1))

            if self.memory_limit:
                self.weight = self.weight.cpu()
                self.filtered_W = torch.zeros_like(self.weight).cuda().to(torch.float16)

            self.filtered_W = self.weight[:,indices_all].clone().to(device)
            
            global indices_list_all
            if self.num == (self.start_num + 1):
                indices_list_all=[]
                
            indices_list_all.append(indices_all)

            self.weight = self.weight.cpu()
        else:
            true_value = x @ self.filtered_W.T
            
        return true_value



class UpGateLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, memory_limit, cpu_only):
        super(UpGateLayer, self).__init__()

        neuron_num = int(weight.size(0) * sparsity)
        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")

        self.weight = weight.clone().to(device)
        self.num = num
        self.memory_limit = memory_limit
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.weight_updated = False

        self.filtered_W = torch.zeros((neuron_num, weight.size(1))).to(torch.float16).to(device)
        
    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")

        if x.size(1)>1:
            self.weight_updated = False
            true_value = x @ self.weight.T.to(device)
            
        else:
            if not self.weight_updated:
                global indices_list_all
                indices = indices_list_all[self.num - (self.start_num + 1)]
                self.filtered_W = self.weight[indices,:].clone().to(device)
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            true_value = x @ self.filtered_W.T
            
        return true_value


def convert_llama_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only):
    
    for name, module in tqdm(model.named_modules(), desc="Convert Llama Models"):
        if "down" in name or "up" in name or "gate" in name:
            num=int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head

                if "down" in name:
                    NewLayer = DownLayer(module.weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only)
                else:
                    NewLayer = UpGateLayer(module.weight, num, sparsity, start_num, memory_limit, cpu_only)
                setattr(parent, attr_name, NewLayer)
                del module

    gc.collect()
    
    return model