
from transformers.siot_variables.siot_improvements import USE_SIOT_IMPROVEMENTS
from transformers.siot_variables.mask_filepath import MASK_FILEPATH
from transformers.siot_variables.use_maskfile import USE_MASKFILE
from transformers.siot_variables.variables import *

def get_used_neurons_count(layer_id: int) -> int:
    return len(get_used_neurons_with_layer_id(layer_id))

def get_used_neurons(layer_name: str, is_loading: bool = False) -> list[int]:
    layer_id = int(layer_name.split(".")[2])
    is_loading_and_up = is_loading and "up" in layer_name
    return get_used_neurons_with_layer_id(layer_id, is_loading_and_up)
    
def get_used_neurons_with_layer_id(layer_id: int, is_loading_and_up: bool = False) -> list[int]:
    if (USE_MASKFILE):
        mask_file = open(MASK_FILEPATH, "r")
        mask_lines = mask_file.readlines()
        
        for line in mask_lines:
            start_of_line = line[:3]
            if (f"{layer_id}:" in start_of_line):
                line_content = line[3:]
                neurons_to_load_of_layer = [int(element) for element in line_content.split(",")]
                break
        return neurons_to_load_of_layer
    
    # TODO: Assumption that layer 0 is proccessed as first element, does not need to be the case
    if (layer_id == 0):
        file = open(current_neurons_always_to_compute_filepath, "w")
        file.write("")
        file.close()
    
    model_neurons_ordered = get_neurons_of_layer_from_file(layer_id, model_neurons_filepath)
    dataset_neurons_ordered = get_neurons_of_layer_from_file(layer_id, dataset_neurons_filepath)
    
    if (layer_id < first_layer or layer_id > last_layer):
        # just use all neurons
        all_neurons = model_neurons_ordered
        all_neurons.sort()
        return all_neurons
    
    if (neurons_always_to_compute_dependency == "model"):
        neurons_always_to_compute_order = model_neurons_ordered
    elif (neurons_always_to_compute_dependency == "dataset"):
        neurons_always_to_compute_order = dataset_neurons_ordered
    else:
        raise RuntimeError("neurons_always_to_compute_dependency set incorrectly")
    
    if (neurons_to_load_dependency == "model"):
        neurons_to_load_order = model_neurons_ordered
    elif (neurons_to_load_dependency == "dataset"):
        neurons_to_load_order = dataset_neurons_ordered
    else:
        raise RuntimeError("neurons_to_load_dependency set incorrectly")
    
    # load all neurons that should be computed always
    index_cut_off = int(len(neurons_always_to_compute_order) * percentage_to_always_compute)
    neurons_always_to_compute = neurons_always_to_compute_order[:index_cut_off]
    neurons_always_to_compute.sort()
    neurons_to_load = neurons_always_to_compute.copy()
    
    # fill up with neurons to load until percentage_to_load is reached
    for neuron in neurons_to_load_order:
        if (neuron not in neurons_to_load):
            neurons_to_load.append(neuron)
        loaded_percentage = len(neurons_to_load) / len(neurons_to_load_order)
        if (loaded_percentage >= percentage_to_load):
            break
    neurons_to_load.sort()
    
    # determine the new indices of the neurons always to compute
    if (is_loading_and_up):
        new_indices_of_neurons_always_to_compute = []
        for new_idx, neuron in enumerate(neurons_to_load):
            if neuron in neurons_always_to_compute:
                new_indices_of_neurons_always_to_compute.append(new_idx)
        
        # write new indices of neurons always to compute
        file_str = f"{layer_id}: "
        new_indices_of_neurons_always_to_compute_str = [str(element) for element in new_indices_of_neurons_always_to_compute]
        file_str += ",".join(new_indices_of_neurons_always_to_compute_str)
        file_str += "\n"
        file = open(current_neurons_always_to_compute_filepath, "a")
        file.write(file_str)
        file.close()
    
    return neurons_to_load
    

def get_neurons_of_layer_from_file(layer_id: int, filepath: str) -> list[int]:
    neurons_file = open(filepath, "r")
    lines = neurons_file.readlines()
    layer_line = lines[layer_id].replace("[", "").replace("]", "")
    neurons_of_layer = [int(element) for element in layer_line.split(",")]
    return neurons_of_layer