# Convert Llama2 Model
import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common
import pickle
from transformers.siot_variables.variables import percentage_overall_to_compute, first_layer, last_layer, percentage_to_always_compute, percentage_to_load
from transformers.siot_variables.use_maskfile import USE_MASKFILE
from transformers.siot_variables.siot_improvements import USE_SIOT_IMPROVEMENTS
indices_list_all = []

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name):
        super(CustomMLPLayer, self).__init__()
        
        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")
        
        if "down" in name:
            neuron_num = int(weight.size(1) * sparsity)
            self.filtered_W = torch.zeros((weight.size(0),neuron_num)).to(torch.float16).to(device)
            loaded_neuron_num = weight.size(1)
        else:
            neuron_num = int(weight.size(0) * sparsity)
            self.filtered_W = torch.zeros((neuron_num, weight.size(1))).to(torch.float16).to(device)
            loaded_neuron_num = weight.size(0)


        self.weight = weight.clone().to(device)
        self.num = num
        self.name = name
        self.token_sparsity = token_sparsity
        self.sparsity = sparsity
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit
        
        self.activation_ratio = 0.0
        self.loaded_neuron_num = loaded_neuron_num
        


    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        global indices_list_all

        if x.size(1)>1:
            self.weight_updated = False
            true_value = x @ self.weight.T.to(device)

            if "down" in self.name:
                squeezed_x = x.clone().squeeze()
                neurons_always_to_compute = self.get_neurons_always_to_compute_from_layer(self.num)
                
                sparsity_stepsize = 0.01
                percentage_overall_to_compute_is_reached = False
                core_neuron_sparsity = (percentage_overall_to_compute - percentage_to_always_compute) / percentage_to_load
                while not percentage_overall_to_compute_is_reached:
                    core_neurons = common.get_core_neurons(squeezed_x, self.token_sparsity, core_neuron_sparsity, self.weight.size(1))
                    neurons_to_compute = list(set(neurons_always_to_compute + core_neurons.tolist()))
                    model_neuron_num = int(self.loaded_neuron_num / percentage_to_load)
                    percentage_to_compute = len(neurons_to_compute) / model_neuron_num
                    percentage_overall_to_compute_is_reached = percentage_to_compute >= percentage_overall_to_compute
                    core_neuron_sparsity += sparsity_stepsize
                    if(core_neuron_sparsity >= 1):
                        break
                
                indices_all = neurons_to_compute
                number_of_neurons = squeezed_x.shape[1]
                self.activation_ratio = len(indices_all) / number_of_neurons
                
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                    self.filtered_W = torch.zeros_like(self.weight).cuda().to(torch.float16)

                self.filtered_W = self.weight[:, indices_all].clone().to(device)
                
                if self.num == (self.start_num + 1):
                    indices_list_all=[]
                    
                indices_list_all.append(indices_all)

                self.weight = self.weight.cpu()
        else:
            if "down" not in self.name:
                if not self.weight_updated:
                    indices = indices_list_all[self.num - (self.start_num + 1)]
                    number_of_neurons = self.weight.shape[0]
                    self.activation_ratio = len(indices) / number_of_neurons
                    self.filtered_W = self.weight[indices,:].clone().to(device)
                    if self.memory_limit:
                        self.weight = self.weight.cpu()
                    self.weight_updated = True

            true_value = x @ self.filtered_W.T
            
        return true_value

    def get_neurons_always_to_compute_from_layer(self, layer_id: int) -> list[int]:
        current_neurons_always_to_compute_filepath = "neurons/current_neurons_always_to_compute.txt"
        file = open(current_neurons_always_to_compute_filepath, "r")
        lines = file.readlines()
        for line in lines:
            start_of_line = line[:3]
            if (f"{layer_id}:" in start_of_line):
                line_content = line[3:]
                neurons_always_to_compute = [int(element) for element in line_content.split(",")]
                break
        return neurons_always_to_compute

def convert_llama_model_siot(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only):
    start_num = first_layer - 1
    end_num = last_layer + 1
    sparsity = percentage_overall_to_compute
    
    if USE_MASKFILE:
        raise RuntimeError("Usage of Filemask needs to be deactivated when using siot Method")
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
                
                NewLayer = CustomMLPLayer(module.weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name)
                setattr(parent, attr_name, NewLayer)
                del module
                custom_layers.append(NewLayer)

    gc.collect()
    model.custom_layers = custom_layers
    
    
    return model