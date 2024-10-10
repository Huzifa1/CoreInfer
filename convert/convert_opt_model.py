import torch.nn as nn
import gc
from scipy.stats import norm
import torch

# Convert Opt Models
class ReduceLayer(nn.Module):
    def __init__(self, weight, bias, sparsity, token_sparsity, memory_limit, name = None):
        super(ReduceLayer, self).__init__()
        remained_neurons = int(weight.size(0) * sparsity)
        
        self.bias = bias.clone()
        self.weight = weight.clone()
        
        self.filtered_W = torch.zeros((remained_neurons, weight.size(1))).to(torch.float16).cpu()
        self.filtered_bias = torch.zeros(remained_neurons).to(torch.float16).cpu()
        self.remained_neurons = remained_neurons
        self.memory_limit = memory_limit
        self.token_sparsity = token_sparsity
    
    def forward(self, x):
        if x.size(0)>1:
            self.weight = self.weight.to(x.device).to(torch.float16)
            self.bias = self.bias.to(x.device).to(torch.float16)
            true_value1 = x @ self.weight.T + self.bias
            true_value = true_value1.clone()

            sorted_values, sorted_indices = torch.sort(true_value1, dim=1, descending=True)
            limit=int(self.token_sparsity*(true_value1>0).sum().item()/true_value1.size(0))
            top_indices = sorted_indices[:, :limit]
            data_flattened = top_indices.reshape(-1)
            unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
            sorted_indices = torch.argsort(counts, descending=True)
            sorted_indices_clu = unique_numbers[sorted_indices]

            indices_all=sorted_indices_clu[:self.remained_neurons].cpu()
            if self.memory_limit:
                self.weight = self.weight.to("cpu")
                self.bias = self.bias.to("cpu")
            
            self.filtered_W = self.weight[indices_all,:].clone().cuda().to(torch.float16)
            self.filtered_bias = self.bias[indices_all].clone().cuda().to(torch.float16)
            global indices_list
            indices_list=indices_all
            
        else:
            true_value = x @ self.filtered_W.T.cuda()+self.filtered_bias.cuda()
            
        return true_value


class ReduceLayer_fc2(nn.Module):
    def __init__(self, weight, bias, sparsity, memory_limit, name = None):
        super(ReduceLayer_fc2, self).__init__()
        self.bias = bias.clone().cpu()
        self.weight = weight.clone().cpu()
        remained_neurons = int(weight.size(1) * sparsity)
        self.filtered_W = torch.zeros((weight.size(0), remained_neurons)).to(torch.float16).cpu()
        self.memory_limit = memory_limit
        
    def forward(self, x):
        if x.size(0)>1:
            self.weight = self.weight.to(x.device).to(torch.float16)
            self.bias = self.bias.to(x.device).to(torch.float16)
            true_value1 = x@self.weight.T+self.bias
            true_value = true_value1.clone()

            global indices_list
            if self.memory_limit:
                self.weight = self.weight.to("cpu")
            self.filtered_W = self.weight[:,indices_list].clone().cuda().to(torch.float16)
            self.bias.cuda()
        else:
            true_value = x @ self.filtered_W.T.cuda()+self.bias.cuda()
            
        return true_value



from tqdm import tqdm

def convert_opt_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit):
    for name, module in tqdm(model.named_modules(), desc="Convert Opt Models"):
        if "fc1" in name:
            num=int(name.split('.')[3])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
                    
                NewLayer = ReduceLayer(module.weight, module.bias, sparsity, token_sparsity, memory_limit, name = name)
                setattr(parent, attr_name, NewLayer)
                del module
        elif "fc2" in name:
            num=int(name.split('.')[3])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
    
                NewLayer = ReduceLayer_fc2(module.weight, module.bias, sparsity, memory_limit, name = name)
                setattr(parent, attr_name, NewLayer)
                del module
                
   
    gc.collect()
    print("Converted Model Done")
    
    return model
 
