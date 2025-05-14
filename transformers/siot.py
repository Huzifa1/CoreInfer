
from transformers.siot_variables.siot_improvements import USE_SIOT_IMPROVEMENTS
from transformers.siot_variables.mask_filepath import MASK_FILEPATH


def get_used_neurons_count(layer_id: int) -> int:
    return len(get_used_neurons_with_layer_id(layer_id))

def get_used_neurons(layer_name: str) -> list[int]:
    layer_id = layer_name.split(".")[2]
    return get_used_neurons_with_layer_id(layer_id)
    
def get_used_neurons_with_layer_id(layer_id: int) -> list[int]:
    mask_file = open(MASK_FILEPATH, "r")
    mask_lines = mask_file.readlines()
    
    for line in mask_lines:
        start_of_line = line[:3]
        if (f"{layer_id}:" in start_of_line):
            line_content = line[3:]
            indices = [int(element) for element in line_content.split(",")]
            break
    return indices