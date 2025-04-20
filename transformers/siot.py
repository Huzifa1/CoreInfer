NEURON_LIMIT = 10

def get_used_neurons_count(layer_id: int) -> int:
    return len(get_used_neurons_with_layer_id(layer_id))

def get_used_neurons(layer_name: str) -> list[int]:
    layer_id = layer_name.split(".")[2]
    return get_used_neurons_with_layer_id(layer_id)
    
def get_used_neurons_with_layer_id(layer_id: int) -> list[int]:
    mask_filepath = ("./masks/25-03-02_cooking_partly.mask")
    mask_file = open(mask_filepath, "r")
    mask_lines = mask_file.readlines()
    
    for line in mask_lines:
        start_of_line = line[:3]
        if (f"{layer_id}:" in start_of_line):
            line_content = line[3:]
            indices = [int(element) for element in line_content.split(",")]
            break
    return indices