# Convert Llama2 Model
import torch.nn as nn
from scipy.stats import norm
import gc
import torch

class DownLayer(nn.Module):
    def __init__(self, weight,num, sparsity, token_sparsity, memory_limit, name = None):
        super(DownLayer, self).__init__()
        self.weight = weight.clone()
        neuron_num = int(weight.size(1)*sparsity)
        self.neuron_num = neuron_num
        self.memory_limit = memory_limit
        if memory_limit:
            self.filtered_W = torch.zeros((weight.size(0),neuron_num)).to(torch.float16).cpu()
        else:
            self.filtered_W = torch.zeros((weight.size(0),neuron_num)).to(torch.float16).cuda()
        self.num = num
        self.token_sparsity = token_sparsity
    def forward(self, x):
        if x.size(1)>1:
            self.weight_updated = False
            x_train=x.clone()
            true_value = x_train@self.weight.T.cuda()
            true_value1 = x.squeeze().clone()

            sorted_values, sorted_indices = torch.sort(true_value1, dim=1, descending=True)
            limit=int(self.token_sparsity*(true_value1>0).sum().item()/true_value1.size(0))
            top_indices = sorted_indices[:, :limit]
            data_flattened = top_indices.reshape(-1)
            unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)
            sorted_indices = torch.argsort(counts, descending=True)
            sorted_indices_clu = unique_numbers[sorted_indices]

            indices_all=sorted_indices_clu[:self.neuron_num].cpu()
            if self.memory_limit:
                self.weight = self.weight.cpu()
                self.filtered_W = torch.zeros_like(self.weight).cuda().to(torch.float16)
                                                                         
            self.filtered_W = self.weight[:,indices_all].clone().cuda()
            
            global indices_list_all
            if self.num ==6:
                indices_list_all=[]
                
            indices_list_all.append(indices_all)

            self.weight = self.weight.cpu()
        else:
            true_value = x @ self.filtered_W.T
            
        return true_value



class UpLayer(nn.Module):
    def __init__(self, weight, num, sparsity, memory_limit, name = None):
        
        super(UpLayer, self).__init__()
        self.weight = weight.clone()
        neuron_num = int(weight.size(0) * sparsity)
        if memory_limit:
            self.filtered_W = torch.zeros((neuron_num, weight.size(1))).to(torch.float16).cpu()
        else:
            self.filtered_W = torch.zeros((neuron_num, weight.size(1))).to(torch.float16).cuda()
        self.num = num
        self.memory_limit = memory_limit
        self.weight_updated = False
        
    def forward(self, x):
        
        if x.size(1)>1:
            self.weight_updated = False
            true_value = x@self.weight.T.cuda()
            
                
        else:
            if not self.weight_updated:
                global indices_list_all
                indices = indices_list_all[self.num-6]
                self.filtered_W = self.weight[indices,:].clone().cuda()
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            true_value = x @ self.filtered_W.T
            
        return true_value


class GateLayer(nn.Module):
    def __init__(self, weight, num, sparsity, memory_limit, name = None):
        super(GateLayer, self).__init__()
        self.weight = weight.clone().cuda()
        neuron_num = int(weight.size(0) * sparsity)
        self.filtered_W = torch.zeros((neuron_num,weight.size(1))).to(torch.float16).cuda()
        self.num = num
        self.memory_limit = memory_limit
        self.weight_updated = False
    def forward(self, x):
        if x.size(1)>1:
            self.weight_updated = False
            true_value = x@self.weight.T.cuda()
            
            
        else:
            if not self.weight_updated:
                global indices_list_all
                indices = indices_list_all[self.num-6]
                self.filtered_W = self.weight[indices,:].clone().cuda()
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            true_value = x @ self.filtered_W.T
        return true_value





from tqdm import tqdm

def convert_llama_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit):
    from tqdm import tqdm
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
    
                NewLayer = DownLayer( module.weight,num, sparsity, token_sparsity, memory_limit, name = name)
                setattr(parent, attr_name, NewLayer)
                del module
                
        elif "up" in name:
            num=int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
    
                NewLayer = UpLayer( module.weight,num, sparsity, memory_limit, name = name)
                setattr(parent, attr_name, NewLayer)
                del module
                
        elif "gate" in name:
            num=int(name.split('.')[2])
            if num>start_num and num<end_num:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else '' # split from right 1 time
                attr_name = name.rsplit('.', 1)[-1]
                if parent_name != '':
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model # for lm_head
    
                NewLayer = GateLayer( module.weight,num,  sparsity, memory_limit, name = name)
                setattr(parent, attr_name, NewLayer)
                del module
    
    # model.cuda()
    gc.collect()
    
    print("Converted Model Done")
    
    return model