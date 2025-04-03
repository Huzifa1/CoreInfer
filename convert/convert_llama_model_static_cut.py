import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common

global_cluster = None

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name):
        super(CustomMLPLayer, self).__init__()
        
        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")
        
        sparsity = 1
        if "down" in name:
            neuron_num = int(weight.size(1) * sparsity)
            self.filtered_W = torch.zeros((weight.size(0),neuron_num)).to(torch.float16).to(device)
        else:
            neuron_num = int(weight.size(0) * sparsity)
            self.filtered_W = torch.zeros((neuron_num, weight.size(1))).to(torch.float16).to(device)


        self.weight = weight.clone().to(device)
        self.num = num
        self.name = name
        self.token_sparsity = token_sparsity
        self.sparsity = sparsity
        self.cpu_only = cpu_only
        self.start_num = start_num
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit


    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        global indices_list_all

        if x.size(1)>1:
            self.weight_updated = False
            true_value = x @ self.weight.T.to(device)

            if "down" in self.name:
                squeezed_x = torch.Tensor(x.clone().squeeze())
                
                # TODO: Make static cut here
                
                number_of_neurons = squeezed_x.shape[1]
                ratio_of_activated_neurons_per_token = list()
                neuron_activation_count = torch.zeros(number_of_neurons)
                overall_activation_after_relu = torch.zeros(number_of_neurons)
                for activation_of_token_raw in squeezed_x:
                    activation_of_token = torch.nn.ReLU()(activation_of_token_raw)
                    overall_activation_after_relu += activation_of_token
                    activated_neurons = (activation_of_token > 0)
                    
                    neuron_activation_count += activated_neurons
                    number_of_activated_neurons_of_token = activated_neurons.sum()
                    ratio_of_activated_neurons = number_of_activated_neurons_of_token / number_of_neurons
                    ratio_of_activated_neurons_per_token.append(ratio_of_activated_neurons)
                    
                mean_ratio_of_activated_neuron = torch.Tensor(ratio_of_activated_neurons_per_token).mean()
                
                cut_off_ratio_to_mean_activation_count = 0.95
                mean_activation_count = float(neuron_activation_count.mean())
                cut_off_activation_count = int(mean_activation_count * cut_off_ratio_to_mean_activation_count)
                over_mean_activation_count_indices = [idx for idx, activation_count in enumerate(neuron_activation_count) if activation_count > cut_off_activation_count]
                number_of_activated_indices = len(over_mean_activation_count_indices)
                ratio_of_activated_neurons = number_of_activated_indices / number_of_neurons
                print("layer {}: activation count sparsity of {}".format(self.num, ratio_of_activated_neurons))
                
                cut_off_ratio_to_mean_activation_value = 0.8
                mean_activation_value = float(overall_activation_after_relu.mean())
                cut_off_activation_value = mean_activation_value * cut_off_ratio_to_mean_activation_value
                over_mean_activation_value_indices = [idx for idx, activation_sum_value in enumerate(overall_activation_after_relu) if activation_sum_value > cut_off_activation_value]
                ratio_of_activated_neurons = len(over_mean_activation_value_indices) / number_of_neurons
                print("layer {}: activation value sparsity of {}".format(self.num, ratio_of_activated_neurons))
                
                
                
                # indices_all = common.get_core_neurons(squeezed_x, token_sparsity, sparsity, self.weight.size(1))
                indices_all = over_mean_activation_count_indices

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
                    self.filtered_W = self.weight[indices,:].clone().to(device)
                    if self.memory_limit:
                        self.weight = self.weight.cpu()
                    self.weight_updated = True

            true_value = x @ self.filtered_W.T
            
        return true_value



def convert_llama_model_static_cut(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only):
    
    start_num = 1
    end_num = 30
    
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

    gc.collect()
    
    return model