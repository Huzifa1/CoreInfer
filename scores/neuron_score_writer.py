
from torch import Tensor

def write_neuron_scores(scores: Tensor, dataset_output_path: str, command_str: str, n_prompts: int):
    file_str = ""
    file_str += command_str
    file_str += f"\nn_prompts: {n_prompts}\n"
    for layer_idx, layer_scores in enumerate(scores):
        file_str += f"{layer_idx}: "
        for neuron_score in layer_scores:
            # rounded_score = round(float(neuron_score), 3)
            file_str += f"{int(neuron_score)};"
        file_str = file_str[:-1]
        file_str += "\n"
    
        output_path = dataset_output_path.replace("_score", "").replace("dataset_run", "scores").replace(".txt", ".score")
        with open(output_path, "a") as output_file:
            output_file.write(file_str)
        file_str = ""