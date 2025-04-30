# Convert Llama2 Model
import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common

from convert.relevant_neurons import get_neuron_scores


indices_list_all = []

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name):
        super(CustomMLPLayer, self).__init__()
        
        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")
        
        if "down" in name:
            neuron_num = int(weight.size(1))
        else:
            neuron_num = int(weight.size(0))

        self.weight = weight.clone().to(device)
        self.num = num
        self.name = name
        self.token_sparsity = token_sparsity
        self.sparsity = sparsity
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit
        
        # self.neuron_scores_list = []
        self.neuron_scores_list = []


    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        global indices_list_all
        
        if (x.size(1)>1 and "down" in self.name):
            squeezed_x = x.clone().squeeze()
            # neuron_scores = get_neuron_scores(squeezed_x)
            # self.neuron_scores_list.append(neuron_scores)
            
            indices_all = common.get_core_neurons(squeezed_x, self.token_sparsity, self.sparsity, self.weight.size(1))
            neuron_scores = torch.zeros(self.neuron_num)
            neuron_scores[indices_all] += 1
            self.neuron_scores_list.append(neuron_scores)
        
        return x @ self.weight.T


def convert_llama_model_score(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only):
    start_num = -1
    end_num = 32
    sparsity = 0.6
    custom_layers = []
    
    for name, module in tqdm(model.named_modules(), desc="Convert Llama Models"):
        if "down" in name:
            num=int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head

                NewLayer = CustomMLPLayer(module.weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name)
                setattr(parent, attr_name, NewLayer)
                del module
                custom_layers.append(NewLayer)

    gc.collect()
    model.custom_layers = custom_layers
    
    return model