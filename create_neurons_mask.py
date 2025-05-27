import json
import torch
import pickle
import os
import transformers.siot_variables.variables as variables

def create_mask(start_num, end_num, model_neurons_percent, loaded_neurons_percent, neurons_filepath, mask_filepath):
    model_neurons, dataset_neurons = load_neurons_tensor(neurons_filepath)
    num_layers = model_neurons.shape[0]
    neurons_num = model_neurons.shape[1]
    
    neurons_to_load = []
    
    for layer_nb in range(num_layers):
        if layer_nb > start_num and layer_nb < end_num:
            # Pick top "model_neurons_percent" from "model_neurons"
            model_neurons_to_load = model_neurons[layer_nb, :round(model_neurons_percent * neurons_num)]
            
            # Now create a new tensor that removes all the indecies picked in model_neurons from dataset_neurons
            mask = ~torch.isin(dataset_neurons[layer_nb], model_neurons_to_load)
            unique_dataset_neurons = dataset_neurons[layer_nb][mask]

            # Now get the rest of neurons to load from unique_dataset_neurons
            dataset_neurons_to_load = unique_dataset_neurons[:round((loaded_neurons_percent - model_neurons_percent) * neurons_num)]      
            neurons_to_load.append(torch.cat((model_neurons_to_load, dataset_neurons_to_load)).tolist())

        else:
            # Load all neurons (0 -> 8192)
            neurons_to_load.append([x for x in range(neurons_num)])
            
    
    with open(mask_filepath, "wb") as f:
        pickle.dump(neurons_to_load, f)
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{current_dir}/transformers/siot_variables/mask_filepath.py", "w") as f:
        f.write(f'MASK_FILEPATH = "{mask_filepath}"')
    
    
def load_neurons_tensor(neurons_filepath):
    with open(neurons_filepath, 'r') as f:
        data = json.load(f)
        
    if "model_neurons" not in data or "dataset_neurons" not in data:
        raise RuntimeError(f"The file: {neurons_filepath} does not have model_neurons or dataset_neurons")
    
    model_neurons = data["model_neurons"]
    dataset_neurons = data["dataset_neurons"]


    return torch.tensor(model_neurons), torch.tensor(dataset_neurons)

def main():
    create_mask(variables.start_num, variables.end_num, variables.model_neurons_percent, variables.loaded_neurons_percent, variables.neurons_filepath, variables.mask_filepath)
    print("Mask is successfully created!")