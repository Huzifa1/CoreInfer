import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common
from convert.relevant_neurons import get_neuron_scores, get_relevant_neuron_indices_dynamic

global_cluster = None

class CustomMLPLayer(nn.Module):
    def __init__(self, weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name, model_modules):
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
        
        self.activation_ratio = 0.0


    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        global indices_list_all

        if x.size(1)>1:
            self.weight_updated = False
            true_value = x @ self.weight.T.to(device)

            if "down" in self.name:
                squeezed_x = torch.Tensor(x.clone().squeeze())
                
                # indices = self.get_most_relevant_neurons_by_with_value(squeezed_x)
                neuron_scores = get_neuron_scores(squeezed_x)
                indices = get_relevant_neuron_indices_dynamic(neuron_scores)
                
                indices_all = torch.Tensor(indices).int().cpu()
                number_of_neurons = squeezed_x.shape[1]
                self.activation_ratio = len(indices_all) / number_of_neurons
                print("layer {}: sparsity of {}".format(self.num, self.activation_ratio))
                

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
    

        
        



def convert_llama_model_dynamic_cut(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only):
    
    start_num = 1
    end_num = 31
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

                NewLayer = CustomMLPLayer(module.weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name, model.named_modules())
                setattr(parent, attr_name, NewLayer)
                del module
                custom_layers.append(NewLayer)

    gc.collect()
    model.custom_layers = custom_layers
    
    return model