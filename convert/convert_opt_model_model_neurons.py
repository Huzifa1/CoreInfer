import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common
import time
# Convert Opt Models
class ReduceLayer(nn.Module):
    def __init__(self, model_neurons, weight, bias, sparsity, token_sparsity, memory_limit, cpu_only, layer_num):
        super(ReduceLayer, self).__init__()
        remained_neurons = int(weight.size(0) * sparsity)
        self.bias = bias.clone()
        self.weight = weight.clone()
        self.filtered_W = torch.zeros((remained_neurons, weight.size(1))).to(torch.float16).cpu()
        self.filtered_bias = torch.zeros(remained_neurons).to(torch.float16).cpu()
        self.remained_neurons = remained_neurons
        self.memory_limit = memory_limit
        self.token_sparsity = token_sparsity
        self.cpu_only = cpu_only
        self.sparsity = sparsity
        self.layer_num = layer_num
        self.model_neurons = model_neurons
        self.adjust_neurons()

    def adjust_neurons(self):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        self.weight = self.weight.to(device).to(torch.float16)
        self.bias = self.bias.to(device).to(torch.float16)
        indices_all = common.get_model_neurons(self.sparsity, self.model_neurons, self.weight.size(0), self.layer_num)

        if self.memory_limit or self.cpu_only:
            self.weight = self.weight.to("cpu")
            self.bias = self.bias.to("cpu")
        
        self.filtered_W = self.weight[indices_all,:].clone().cpu().to(device).to(torch.float16)
        self.filtered_bias = self.bias[indices_all].clone().cpu().to(device).to(torch.float16)
        global indices_list
        indices_list=indices_all

    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        if x.size(0)>1:
            self.weight = self.weight.to(x.device).to(torch.float16)
            self.bias = self.bias.to(x.device).to(torch.float16)
            true_value = x @ self.weight.T + self.bias
        else:
            true_value = x @ self.filtered_W.T.to(device) + self.filtered_bias.to(device)  
        return true_value


class ReduceLayer_fc2(nn.Module):
    def __init__(self, weight, bias, sparsity, memory_limit, cpu_only, layer_num):
        super(ReduceLayer_fc2, self).__init__()
        self.bias = bias.clone().cpu()
        self.weight = weight.clone().cpu()
        remained_neurons = int(weight.size(1) * sparsity)
        self.filtered_W = torch.zeros((weight.size(0), remained_neurons)).to(torch.float16).cpu()
        self.memory_limit = memory_limit
        self.cpu_only = cpu_only
        self.layer_num = layer_num
        self.adjust_neurons()

    def adjust_neurons(self):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        self.weight = self.weight.to(device).to(torch.float16)
        self.bias = self.bias.to(device).to(torch.float16)

        global indices_list
        if self.memory_limit:
            self.weight = self.weight.to("cpu")

        self.filtered_W = self.weight[:,indices_list].clone().to(device).to(torch.float16)
        self.bias.to(device)
        
    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        if x.size(0)>1:
            self.weight = self.weight.to(x.device).to(torch.float16)
            self.bias = self.bias.to(x.device).to(torch.float16)
            true_value = x@self.weight.T+self.bias
        else:
            true_value = x @ self.filtered_W.T.to(device) + self.bias.to(device)
        return true_value

def load_model_neurons_tensor():
    file = "/local/artemb/CoreInfer/convert/model_neurons_opt.txt"
    with open(file, "r") as f:
        lines = f.readlines()
        data = [list(map(int, line.strip().strip("[]").split(","))) for line in lines]

    max_len = max(len(row) for row in data)
    padded = [row + [-1]*(max_len - len(row)) for row in data]
    return torch.tensor(padded, dtype=torch.long).contiguous()

def convert_opt_model_model_neurons(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only):
    global model_neurons
    model_neurons = load_model_neurons_tensor()
    for name, module in tqdm(model.named_modules(), desc="Convert Opt Models"):
        if "fc1" in name or "fc2" in name:
            num=int(name.split('.')[3])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
                    
                if "fc1" in name:
                    NewLayer = ReduceLayer(model_neurons, module.weight, module.bias, sparsity, token_sparsity, memory_limit, cpu_only, num)
                else:
                    NewLayer = ReduceLayer_fc2(module.weight, module.bias, sparsity, memory_limit, cpu_only, num)

                setattr(parent, attr_name, NewLayer)
                del module
                
    gc.collect()
    
    return model
 
