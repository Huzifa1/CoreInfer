n_layer = 28
first_layer = 5
last_layer = 25
percentage_to_load = 0.6
percentage_to_always_compute = 0.25
percentage_overall_to_compute = 0.4

# Use "dataset" or "model" neurons for neurons always to compute or to load
neurons_always_to_compute_dependency = "model"
neurons_to_load_dependency = "dataset"


model_neurons_filepath = "neurons/model_neurons_llama3.txt"
dataset_neurons_filepath = "neurons/model_neurons_llama3_2.txt"

current_neurons_always_to_compute_filepath = "neurons/current_neurons_always_to_compute.txt"