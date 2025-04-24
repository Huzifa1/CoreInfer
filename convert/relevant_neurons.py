
import torch


def get_neuron_scores(x: torch.Tensor):
    value_weight = 0.5
    count_weight = 0.5
    return get_neuron_score_with_value_and_frequency(x, value_weight, count_weight)

def get_neuron_score_with_value_and_frequency(x: torch.Tensor, sum_weight: float, count_weight: float) -> torch.Tensor:
    # Set negative values to 0
    x_relu = x * (x > 0)
    # Sum activation values of all tokens for each neuron
    x_summed = x_relu.sum(dim=0)
    # Get highest activation sum
    top_k_to_use = 100
    highest_values, highest_indices = torch.topk(x_summed, top_k_to_use)
    highest_sum_score = highest_values[-1]
    # Normalize over the highest score
    activations_sum_scores = x_summed / highest_sum_score

    # Get the count of how many times did each neuron got activated (value > 0)
    x_activation_count = (x > 0).sum(dim=0)
    # Get highest activation sum
    top_k_to_use = 100
    highest_values, highest_indices = torch.topk(x_activation_count, top_k_to_use)
    highest_count_score = highest_values[-1]
    # Normalize over the highest score
    activations_count_scores = x_activation_count / highest_count_score

    # Now get total score for each neuron
    total_scores = activations_sum_scores * sum_weight + activations_count_scores * count_weight
    return total_scores








def get_relevant_neuron_indices_static(neuron_scores: torch.Tensor, cut_off_ratio_static: float):
    sorted_values, sorted_indices = neuron_scores.sort(descending=True)
    limit_index = int(len(neuron_scores) * cut_off_ratio_static)
    return sorted_indices[:limit_index]

def get_relevant_neuron_indices_moving(neuron_scores: torch.Tensor, activation_ratios: list[float], layer_num: int):
    sorted_values, sorted_indices = neuron_scores.sort(descending=True)
    limit_index = int(len(neuron_scores) * activation_ratios[layer_num])
    return sorted_indices[:limit_index]

def get_relevant_neuron_indices_dynamic(neuron_scores: torch.Tensor):
    cut_off_ratio_to_mean_score = 0.8
    cut_off_score = 0.34
    minimum_used_ratio = 0.6
    return get_relevant_neuron_indices_by_score(neuron_scores, cut_off_score, minimum_used_ratio)

def get_relevant_neuron_indices_by_score(neuron_scores: torch.Tensor, cut_off_score: float, minimum_used_ratio: float):
    score_sum = neuron_scores.sum()
    minimum_used_ratio_is_reached = False
    while not minimum_used_ratio_is_reached:
        over_contribution_indices = [idx for idx, score in enumerate(neuron_scores) if score > cut_off_score]
        used_ratio = len(over_contribution_indices) / len(neuron_scores)
        minimum_used_ratio_is_reached = (used_ratio > minimum_used_ratio)
        if not minimum_used_ratio_is_reached:
            cut_off_score -= 0.01
    return over_contribution_indices
    

def get_relevant_neuron_indices_by_mean(neuron_scores: torch.Tensor, cut_off_ratio_to_mean_score: float):
    mean_score = torch.Tensor(neuron_scores).mean()
    cut_off_score = mean_score * cut_off_ratio_to_mean_score
    over_mean_activation_value_indices = [idx for idx, score in enumerate(neuron_scores) if score >= cut_off_score]
    return over_mean_activation_value_indices











def get_neurons_by_means(squeezed_x):
    number_of_neurons = squeezed_x.shape[1]
    ratio_of_activated_neurons_per_token = list()
    neuron_activation_count = torch.zeros(number_of_neurons)
    overall_activation_after_relu = torch.zeros(number_of_neurons)
    for activation_of_token_raw in squeezed_x:
        activation_of_token = torch.nn.ReLU()(activation_of_token_raw)
        overall_activation_after_relu += activation_of_token
        activated_neurons = (activation_of_token > 0)
        
        neuron_activation_count += activated_neurons
        number_of_activated_neurons_of_token = activated_neurons.sum()
        ratio_of_activated_neurons = number_of_activated_neurons_of_token / number_of_neurons
        ratio_of_activated_neurons_per_token.append(ratio_of_activated_neurons)
        
    mean_ratio_of_activated_neuron = torch.Tensor(ratio_of_activated_neurons_per_token).mean()
    
    cut_off_ratio_to_mean_activation_count = 0.8
    mean_activation_count = float(neuron_activation_count.mean())
    cut_off_activation_count = int(mean_activation_count * cut_off_ratio_to_mean_activation_count)
    over_mean_activation_count_indices = [idx for idx, activation_count in enumerate(neuron_activation_count) if activation_count > cut_off_activation_count]
    number_of_activated_indices = len(over_mean_activation_count_indices)
    ratio_of_activated_neurons = number_of_activated_indices / number_of_neurons
    # print("layer {}: activation count sparsity of {}".format(self.num, ratio_of_activated_neurons))
    
    cut_off_ratio_to_mean_activation_value = 0.95
    mean_activation_value = float(overall_activation_after_relu.mean())
    cut_off_activation_value = (mean_activation_value + overall_activation_after_relu.min().abs()) * cut_off_ratio_to_mean_activation_value
    over_mean_activation_value_indices = [idx for idx, activation_sum_value in enumerate(overall_activation_after_relu) if activation_sum_value >= cut_off_activation_value]
    ratio_of_activated_neurons = len(over_mean_activation_value_indices) / number_of_neurons
    # print("layer {}: activation value sparsity of {}".format(self.num, ratio_of_activated_neurons))
    
    unique_indices_combined = list(set(over_mean_activation_count_indices) & set(over_mean_activation_value_indices))
    ratio_of_activated_neurons = len(unique_indices_combined) / number_of_neurons
    # print("layer {}: combined sparsity of {}".format(self.num, ratio_of_activated_neurons))
    
    # indices_all = common.get_core_neurons(squeezed_x, token_sparsity, sparsity, self.weight.size(1))
    indices = over_mean_activation_value_indices
    return indices





def get_mean_average_activation_of_file(filepath: str, number_of_modified_layers: int):
    with open(filepath) as file:
        content = file.readlines()
    
    mean_activation_ratios = list()
    for line in content:
        if ("Mean activation ratio: " in line):
            mean_activation_ratio = float(line.replace("Mean activation ratio: ", ""))
            mean_activation_ratios.append(mean_activation_ratio)
    
    overall_mean_activation_ratio_modified_layers = float(torch.Tensor(mean_activation_ratios).mean())
    
    number_of_layers = 32
    overall_mean_activation_ratio = (number_of_modified_layers * overall_mean_activation_ratio_modified_layers + number_of_layers - number_of_modified_layers) / number_of_layers
    return overall_mean_activation_ratio

def get_activation_ratios_of_layermask(filepath: str):
    with open(filepath) as file:
        content = file.readlines()
    
    activation_ratios = list()
    for line in content:
        if ("sparsity " in line):
            sparsity_str = line.split("sparsity ")[-1].replace("\n", "")
            if ("nan" in sparsity_str):
                activation_ratios.append(1)
            else:
                activation_ratios.append(float(sparsity_str))
    return activation_ratios

def get_mean_acitvation_ratio_of_layermask(filepath: str):
    activation_ratios = get_activation_ratios_of_layermask(filepath)
    mean_activation_ratio = float(torch.Tensor(activation_ratios).mean())
    return mean_activation_ratio