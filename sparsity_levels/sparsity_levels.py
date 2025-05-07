import pickle
import sys
import os
sys.path.append("/".join(os.getcwd().split("/")[:-1]))
import common
import utils
import json
import re
import numpy as np

def get_sparsity_levels_params(model_name, num_layers, activations, max_iterations, number_of_requested_values, sparsity_goal, allowed_tolerance, start_num, end_num):

    counter = 0
    possible_values = []
    possible_sparsity_levels = []

    possible_weights = range(10, 100, 5)
    possible_thresholds = range(15, 51, 1)

    calculated_sparsity_levels = []

    while counter < max_iterations and len(possible_values) < number_of_requested_values:
        i = 0
        for sum_weight in possible_weights:
            sum_weight = sum_weight / 100
            count_weight = 1 - sum_weight
            skip_thresholds = False
            for threshold in possible_thresholds:
                threshold = threshold / 100
                
                if skip_thresholds:
                    continue
                
                if (
                    (sum_weight < 0.3 or count_weight < 0.3) and threshold < 0.45 or
                    (sum_weight < 0.4 or count_weight < 0.4) and threshold < 0.35 or
                    (sum_weight < 0.6 or count_weight < 0.6) and threshold < 0.25
                ):
                    continue
                
                top_k = 100
                params = [sum_weight, count_weight, threshold, top_k]
            
                # Get sparsity levels
                if counter == 0:
                    sparsity_levels = common.get_sparsity_levels(model_name, num_layers, params[0], params[1], params[2], params[3], activations)
                    calculated_sparsity_levels.append(sparsity_levels)
                else:
                    sparsity_levels = calculated_sparsity_levels[i]
                
                # Get the avg sparsity
                cut_sparsity_levels = sparsity_levels[start_num+1:end_num]
                avg_sparsity = sum(cut_sparsity_levels) / len(cut_sparsity_levels)
                
                            
                print(f"{i}. Actual sparsity: {avg_sparsity} | Params: {(params[0], params[1], params[2], params[3])}")
                
                if avg_sparsity < sparsity_goal - allowed_tolerance:
                    skip_thresholds = True
                                    
                if abs(sparsity_goal - avg_sparsity) <= allowed_tolerance:
                    print(f"Found a possible params: {params}")
                    possible_values.append(params)
                    possible_sparsity_levels.append(sparsity_levels)
                
                if counter > max_iterations or len(possible_values) >= number_of_requested_values:
                    break
                
                i += 1
                
            if counter > max_iterations or len(possible_values) >= number_of_requested_values:
                break
            
        counter += 1
        
    return possible_values, possible_sparsity_levels

def save_levels(levels, dataset_name, dir_path):
    levels_path = f"{dir_path}/{dataset_name}/"
    os.makedirs(os.path.dirname(levels_path), exist_ok=True)
    for i,l in enumerate(levels):
        with open(f"{levels_path}sparsity_levels{i}.pkl", "wb") as f:
            pickle.dump(l, f)

def draw_levels(levels, title, labels = None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Create the plot
    plt.figure(figsize=(10, 6))
    # Create a colormap with distinct colors using HSV color space
    num_configs = len(levels)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_configs))

    # Create different line styles to alternate between
    line_styles = ['-', '--', '-.', ':']
    line_widths = [1, 1.5, 2]

    # Create the plot with different combinations of colors, styles and widths
    for i in range(num_configs):
        style_idx = i % len(line_styles)
        width_idx = (i // len(line_styles)) % len(line_widths)
        
        if labels:
            label = labels[i]
        else:
            label = f'Sparsity_levels{i}'
        
        plt.plot(list(range(6, 27)), 
                levels[i][6:27],
                linestyle=line_styles[style_idx],
                color=colors[i],
                linewidth=line_widths[width_idx],
                label=label)

    # Add labels and title
    plt.xlabel('Layer Number')
    plt.xticks(range(6, 27, 1))
    plt.ylabel('Sparsity Level')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8)


    # Save the figure
    plt.tight_layout()
    plt.show()

def load_sparsity_levels(dataset_name, dir_path):
    sparsity_levels = []
    dir_path = f"{dir_path}/{dataset_name}"
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, "rb") as f:
            levels = pickle.load(f)
            sparsity_levels.append(levels)
            
    return sparsity_levels

def print_levels_results(results_path):
    datasets_dir = os.listdir(results_path)
    output = {}
    
    for dataset_name in datasets_dir:
        # print(f"\nTask: {dataset_name}")
        output[dataset_name] = {}
        full_path = os.path.join(results_path, dataset_name)
        for results_file in sorted(os.listdir(full_path)):
            results_file_path = os.path.join(full_path, results_file)
            with open(results_file_path, 'r') as f:
                results = json.load(f)
                temp = results[dataset_name]
                i = 0
                for key,value in temp.items():
                    if i == 1:
                        # print(f"{results_file} ({metric_key}): {value}")
                        sparsity_level_key = re.search(r'sparsity_levels\d+\.pkl', results_file).group(0)
                        output[dataset_name][sparsity_level_key] = value
                        break
                    i += 1
                
    return output
        
        

def print_actual_sparsity(model_name, path_to_file, start_num = 5, end_num = 27):

    file_output = utils.read_indecies_file(path_to_file, model_name, start_num, end_num)

    output = {}
    results = {}
    for o in file_output:
        task_name = o["task_name"]
        if task_name not in results:
            results[task_name] = []
            
        results[task_name].append({
            'sparsity_name': o['sparsity_name'],
            'avg': o['avg']
        })
        
    for key,value in results.items():
        value.sort(key=lambda x: x['sparsity_name'])
        # print(f"\nTask: {key}")
        output[key] = {}
        for v in value:
            output[key][v['sparsity_name']] = v['avg']
            # print(f"{v['sparsity_name']}: {v['avg']}")
            
    return output


def create_md_table_for_results(model_name, results_path, indecies_all_path):
    results = print_levels_results(results_path)
    actual_sparsity = print_actual_sparsity(model_name, indecies_all_path)
    
    tables = {}
    for task,sparsity_levels in results.items():
        tables[task] = []
        for level,value in sparsity_levels.items():
            tables[task].append([level, round(actual_sparsity[task][level], 3), round(value, 3)])
    
    for task, rows in tables.items():
        header = ["Levels", "Actual Sparsity", "Metric Value"]
        markdown = "| " + " | ".join(header) + " |\n"
        markdown += "| " + " | ".join("---" for _ in header) + " |\n"

        # Add data rows
        for row in rows:
            markdown += "| " + " | ".join(str(item) for item in row) + " |\n"

        print(f"### {task}\n")
        print(markdown)
        

def calculate_avg_levels(dataset_name, sparsity_levels_dir_path):
    sparsity_levels = load_sparsity_levels(dataset_name, sparsity_levels_dir_path)
    sparsity_levels_array = np.array(sparsity_levels)
    avg_levels = np.mean(sparsity_levels_array, axis=0)

    return avg_levels