# Convert Llama2 Model
import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common
import pickle
from transformers.siot_variables.siot_improvements import USE_SIOT_IMPROVEMENTS

indices_list_all = []

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name, original_neurons_num, siot_method_config):
        super(CustomMLPLayer, self).__init__()
        
        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")
        
        neuron_num = round(original_neurons_num * sparsity)
        if "down" in name:
            loaded_neuron_num = weight.size(1)
        else:
            loaded_neuron_num = weight.size(0)

        if neuron_num > loaded_neuron_num:
            raise RuntimeError(f"Number of required neurons ({neuron_num}) is larger than the number of loaded neurons ({loaded_neuron_num})")
        
        if siot_method_config["base_neurons_percent"] > sparsity:
            p = siot_method_config["base_neurons_percent"]
            raise RuntimeError(f"base_neurons_percent ({p}) is larger than sparsity ({sparsity}).")

        self.weight = weight.to(device)
        self.num = num
        self.name = name
        self.token_sparsity = token_sparsity
        self.sparsity = sparsity
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit
        self.original_neurons_num = original_neurons_num        
        self.loaded_neuron_num = loaded_neuron_num
        self.base_neurons_percent = siot_method_config["base_neurons_percent"]

    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        global indices_list_all
        if x.size(1)>1:
            self.weight_updated = False
            true_value = x @ self.weight.T.to(device)
            num_neurons = self.weight.size(1)
            
            if self.memory_limit:
                self.weight = self.weight.cpu()

            if "down" in self.name:
                squeezed_x = x.squeeze()

                # Get base neurons
                # This works since when loading, base neurons are sorted at the beginning
                base_neuron_num = int(self.base_neurons_percent * self.original_neurons_num)
                base_neurons = torch.arange(0, base_neuron_num)
                
                if self.base_neurons_percent < self.sparsity:
                    # Now fill up with core neurons
                    core_neurons = common.get_core_neurons(squeezed_x, self.token_sparsity, 1, num_neurons)
                
                    # Now remove the overlap
                    mask = ~torch.isin(core_neurons, base_neurons)
                    unique_core_neurons = core_neurons[mask]
                    
                    # Now get the rest of neurons to load from unique_core_neurons
                    unique_core_neurons_to_compute = unique_core_neurons[:int((self.sparsity - self.base_neurons_percent) * self.original_neurons_num)]      
                    indices_all = torch.cat((base_neurons, unique_core_neurons_to_compute))
                else:
                    indices_all = base_neurons
                
                if self.num == (self.start_num + 1):
                    indices_list_all=[]
                    
                indices_list_all.append(indices_all)

        else:
            indices = indices_list_all[self.num - (self.start_num + 1)]

            if "down" in self.name:
                true_value = x @ self.weight[:, indices].T.to(device)
            else:
                true_value = x @ self.weight[indices, :].T.to(device)
            
        return true_value


def convert_llama_model_siot(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, original_neurons_num, siot_method_config):
    
    if not USE_SIOT_IMPROVEMENTS:
        raise RuntimeError("SIOT Improvements / partial loading needs to be activated for siot Method")
    
    custom_layers = []
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
                
                NewLayer = CustomMLPLayer(module.weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name, original_neurons_num, siot_method_config)
                setattr(parent, attr_name, NewLayer)
                del module
                custom_layers.append(NewLayer)

    gc.collect()
    model.custom_layers = custom_layers
    
    
    return model