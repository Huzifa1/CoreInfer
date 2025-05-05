USE_SIOT_IMPROVEMENTS = False
MASK_FILEPATH = "masks/scores_2025_04_27_13_11_0.7_4_26.mask"


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