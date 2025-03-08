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

        # If we are in the prefilling stage
        if x.size(1)>1:
            # Assume x of size [1, 160, 14336], number of tokens in the prompt = 160
            self.weight_updated = False
            x_train=x.clone()
            true_value = x_train@self.weight.T
            # true_value1 is of size [160, 14336]
            true_value1 = x.squeeze().clone()

            # Sort the values in descending order based on their activation values
            # sorted_indices is of size [160, 14336]
            sorted_values, sorted_indices = torch.sort(true_value1, dim=1, descending=True)

            # This will check the number of positive activation in true_value1 and then normalize 
            # it by the number of tokens, 160 in this case, then select only `token_sparsity` of them
            # Assume limit is 1500
            limit=int(self.token_sparsity*(true_value1>0).sum().item()/true_value1.size(0))

            # Select the top `limit` indices for each token
            # top_indices is of size [160, 1500]
            top_indices = sorted_indices[:, :limit]

            # Flatten the top_indices to be 1D array, it will be of size [160*1500]=[240000]
            data_flattened = top_indices.reshape(-1)

            # Since now that top_indices is flattened to 1D array, then some indices will be repeated for different tokens
            # So then we count how many times each index is repeated
            unique_numbers, counts = data_flattened.unique(return_counts=True, sorted=True)

            # This will sort the `counts` and then return the indices of the sorted counts
            # For example, sorted_indices[0] will be the index to the `count` array of the number that is repeated the most
            sorted_indices = torch.argsort(counts, descending=True)
            # This will contain the indices of the unique numbers sorted
            # For example, sorted_indices_clu[0] will be the index to the neuron that is repeated the most
            sorted_indices_clu = unique_numbers[sorted_indices]

            # Select top `neuron_num (beta)` neurons
            # This is basically the indices of the core neurons
            indices_all=sorted_indices_clu[:self.neuron_num].cpu()
            
            # If memory_limit is True, then we will store the filtered_W in CPU
            if self.memory_limit:
                self.weight = self.weight.cpu()
                self.filtered_W = torch.zeros_like(self.weight).to(torch.float16)
                                                             
            self.filtered_W = self.weight[:,indices_all].clone()
            
            global indices_list_all
            # This is not so accurate, we should pass the `start_num` as an argument
            # This assumes the default value for `start_num`, which is 5
            # Which means that the first layer to have this functionality is layer 6
            if self.num ==6:
                # If this is the first layer to have this functionality, then we will initialize the indices_list_all
                indices_list_all=[]
            
            # Append the indices of the "core neurons" to the global list
            indices_list_all.append(indices_all)

            self.weight = self.weight.cpu()
        # We are in the decoding stage
        else:
            # Compute using only the core neurons
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
        # If we are in the prefilling stage
        if x.size(1)>1:
            # Compute normally using all neurons
            self.weight_updated = False
            true_value = x@self.weight.T
            
                
        else:
            # If this is the first time we are in the decoding stage
            if not self.weight_updated:
                global indices_list_all
                # Get the indices of the core neurons from the "DownLayer"
                indices = indices_list_all[self.num-6]
                # Set the filtered_W to be the core neurons only
                self.filtered_W = self.weight[indices,:].clone()
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            # Compute using only the core neurons
            true_value = x @ self.filtered_W.T
            
        return true_value


class GateLayer(nn.Module):
    # This is exactly the same as the "UpLayer"
    def __init__(self, weight, num, sparsity, memory_limit, name = None):
        super(GateLayer, self).__init__()
        self.weight = weight.clone()
        neuron_num = int(weight.size(0) * sparsity)
        self.filtered_W = torch.zeros((neuron_num,weight.size(1))).to(torch.float16)
        self.num = num
        self.memory_limit = memory_limit
        self.weight_updated = False
    def forward(self, x):
        if x.size(1)>1:
            self.weight_updated = False
            true_value = x@self.weight.T
            
        else:
            if not self.weight_updated:
                global indices_list_all
                indices = indices_list_all[self.num-6]
                self.filtered_W = self.weight[indices,:].clone()
                if self.memory_limit:
                    self.weight = self.weight.cpu()
                self.weight_updated = True
            true_value = x @ self.filtered_W.T
        return true_value





from tqdm import tqdm

def convert_llama_model(model, sparsity, start_num, end_num, token_sparsity, memory_limit):
    from tqdm import tqdm
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
                if "down" in name:
                    NewLayer = DownLayer( module.weight,num, sparsity, token_sparsity, memory_limit, name = name)
                elif "up" in name:
                    NewLayer = UpLayer( module.weight,num, sparsity, memory_limit, name = name)
                elif "gate" in name:
                    NewLayer = GateLayer( module.weight,num,  sparsity, memory_limit, name = name)
                setattr(parent, attr_name, NewLayer)
                del module
    
    # model.cuda()
    gc.collect()
    
    print("Converted Model Done")
    
    return model