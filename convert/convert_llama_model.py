# Convert Llama2 Model
import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common
import pickle

indices_list_all = []
previous_indices_list_all = []

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, name):
        super(CustomMLPLayer, self).__init__()

        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")

        if "down" in name:
            neuron_num = int(weight.size(1) * sparsity)
        else:
            neuron_num = int(weight.size(0) * sparsity)


        self.weight = weight.contiguous().to(device)
        self.num = num
        self.name = name
        self.token_sparsity = token_sparsity
        self.sparsity = sparsity
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.end_num = end_num
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit
        self.is_reorderd = False
     
    def forward(self, x):
        global indices_list_all, previous_indices_list_all

        if x.size(1)>1:
            self.weight_updated = False
            
            if self.is_reorderd:
                indices = previous_indices_list_all[self.num - (self.start_num + 1)]
                common.reorder_tensor(self.weight, indices, is_reverse="down" not in self.name, is_restore=True)
            
            true_value = x @ self.weight.T

            if "down" in self.name:
                squeezed_x = x.squeeze()
                indices_all = common.get_core_neurons(squeezed_x, self.token_sparsity, self.sparsity, self.weight.size(1))
                
                common.reorder_tensor(self.weight, indices_all)
                self.is_reorderd = True
                end_index = indices_all.shape[0]
                self.filtered_W = self.weight[:, 0:end_index]
                
                if self.num == (self.start_num + 1):
                    indices_list_all = []
                        
                indices_list_all.append(indices_all)
                
                if self.num == self.end_num - 1:
                    previous_indices_list_all = indices_list_all.copy()
 

        else:
            if "down" not in self.name:
                if not self.weight_updated:
                    indices = indices_list_all[self.num - (self.start_num + 1)]
                    common.reorder_tensor(self.weight, indices, is_reverse=True)
                    self.is_reorderd = True
                    end_index = indices.shape[0]
                    self.filtered_W = self.weight[0:end_index, :]
                    self.weight_updated = True
                    
            true_value = x @ self.filtered_W.T
        return true_value


def convert_llama_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, sparsity_levels_path):
    custom_layers = []
    
    sparsity_levels = None
    if sparsity_levels_path is not None:
        with open(sparsity_levels_path, 'rb') as f:
            sparsity_levels = pickle.load(f)

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
                    
                if sparsity_levels is not None:
                    sparsity = sparsity_levels[num]

                NewLayer = CustomMLPLayer(module.weight, num, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, name)
                setattr(parent, attr_name, NewLayer)
                del module
                custom_layers.append(NewLayer)

    gc.collect()
    model.custom_layers = custom_layers
    return model