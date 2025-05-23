# Convert Llama2 Model
import torch.nn as nn
import gc
import torch
from tqdm import tqdm
import common

indices_list_all = []

class CustomMLPLayer(nn.Module):
    def __init__(self, model_neurons, weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name, hybrid_split):
        super(CustomMLPLayer, self).__init__()

        device = torch.device("cpu") if memory_limit or cpu_only else torch.device("cuda")

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
        self.model_neurons = model_neurons[num]
        self.hybrid_split = hybrid_split

    def forward(self, x):
        device = torch.device("cpu") if self.cpu_only else torch.device("cuda")
        global indices_list_all

        if x.size(1)>1:
            self.weight_updated = False
            true_value = x @ self.weight.T.to(device)

            if "down" in self.name:
                squeezed_x = x.clone().squeeze()
                core_indices = common.get_core_neurons(squeezed_x, self.token_sparsity, self.sparsity - (self.sparsity*self.hybrid_split), self.weight.size(1))
                split_model_indices = self.model_neurons[:int(self.sparsity*self.hybrid_split*self.weight.size(1))]
                indices_all = torch.cat([split_model_indices, core_indices]).unique(sorted=True)
                
                # Ensure indices_all has at least shape self.sparsity
                target_size = int(self.weight.size(1) * self.sparsity)
                if len(indices_all) < target_size:
                    remaining_model_indecies = self.model_neurons[len(split_model_indices):]
                    mask = ~torch.isin(remaining_model_indecies, indices_all)
                    to_add = remaining_model_indecies[mask]
                    needed = target_size - len(indices_all)
                    indices_all = torch.cat((indices_all, to_add[:needed]))

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

def load_model_neurons_tensor():
    file = "/local/artemb/CoreInfer/convert/model_neurons_llama3.txt"
    with open(file, "r") as f:
        lines = f.readlines()
        data = [list(map(int, line.strip().strip("[]").split(","))) for line in lines]

    max_len = max(len(row) for row in data)
    padded = [row + [-1]*(max_len - len(row)) for row in data]
    return torch.tensor(padded, dtype=torch.long).contiguous()

def convert_llama_model_hybrid_neurons(model, sparsity, start_num, end_num, token_sparsity, memory_limit, cpu_only, hybrid_split):
    global model_neurons
    model_neurons = load_model_neurons_tensor()
    c = 0
    for name, module in tqdm(model.named_modules(), desc="Convert Llama Models"):
        c += 1
        if "down" in name or "up" in name or "gate" in name:
            num=int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head

                NewLayer = CustomMLPLayer(model_neurons, module.weight, num, sparsity, start_num, token_sparsity, memory_limit, cpu_only, name, hybrid_split)
                setattr(parent, attr_name, NewLayer)
                del module

    gc.collect()
    print(c)
    return model
