import json
import torch
import pickle
import os
import transformers.siot_variables.variables as variables

current_dir = os.path.dirname(os.path.abspath(__file__))

def create_mask(start_num, end_num, base_neurons_percent, base_neurons_type, loaded_neurons_percent, model_neurons_filepath, dataset_neurons_filepath, mask_filepath):
    model_neurons = load_neurons_tensor(model_neurons_filepath)
    dataset_neurons = load_neurons_tensor(dataset_neurons_filepath)
    
    base_neurons = model_neurons if base_neurons_type == "model" else dataset_neurons
    secondary_neurons = dataset_neurons if base_neurons_type == "model" else model_neurons
    
    
    num_layers = model_neurons.shape[0]
    neurons_num = model_neurons.shape[1]
    
    neurons_to_load = []
    
    for layer_nb in range(num_layers):
        if layer_nb > start_num and layer_nb < end_num:
            # Pick top "base_neurons_percent" from "base_neurons"
            base_neurons_to_load = base_neurons[layer_nb, :round(base_neurons_percent * neurons_num)]
            
            # Now create a new tensor that removes all the indecies picked in base_neurons from secondary_neurons
            mask = ~torch.isin(secondary_neurons[layer_nb], base_neurons_to_load)
            unique_secondary_neurons = secondary_neurons[layer_nb][mask]

            # Now get the rest of neurons to load from unique_secondary_neurons
            secondary_neurons_to_load = unique_secondary_neurons[:round((loaded_neurons_percent - base_neurons_percent) * neurons_num)]      
            neurons_to_load.append(torch.cat((base_neurons_to_load, secondary_neurons_to_load)).tolist())

        else:
            # Load all neurons (0 -> 8192)
            neurons_to_load.append([x for x in range(neurons_num)])
            
    
    with open(f"{current_dir}/{mask_filepath}", "wb") as f:
        pickle.dump(neurons_to_load, f)
        
    with open(f"{current_dir}/transformers/siot_variables/mask_filepath.py", "w") as f:
        f.write(f'MASK_FILEPATH = "{current_dir}/{mask_filepath}"')
    
    
def load_neurons_tensor(neurons_filepath):
    with open(f"{current_dir}/{neurons_filepath}", 'r') as f:
        data = json.load(f)
        
    if "neurons" not in data:
        raise KeyError(f"The file: {neurons_filepath}/{neurons_filepath} does not key 'data'")
    
    neurons = data["neurons"]

    return torch.tensor(neurons)

def main():
    if variables.base_neurons_type not in ["model", "dataset"]:
        raise RuntimeError("base_neurons_type must be either set to 'model' or 'dataset'")
    
    create_mask(variables.start_num, variables.end_num, variables.base_neurons_percent, variables.base_neurons_type, variables.loaded_neurons_percent, variables.model_neurons_filepath, variables.dataset_neurons_filepath, variables.mask_filepath)
    print("Mask is successfully created!")