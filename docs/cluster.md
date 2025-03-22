# Explaining `cluster.py`:

This will explain the flow of running inference using the `llama` model.

## `main` function:
- First it calls `_load_model_data`. It's very similar to `_load_model` in the `coerinfer.py` file.
- Then it calls `process_data`. It just prepares an array of the datasets, currently the function only supports the `truthful_qa` and `wmt16-de-en` datasets.
- Then it calls `cluster_llama_data`.
- Then it calls `save_cluster`.
- Finally, it calls `save_llama_neurons`.

## `cluster_llama_data` function:
- In summary, it gets the core neuron for each sentence in the dataset.
- For more details, check the comments in the code.

## `save_cluster` function:
- This function uses the **elbow** method to find the best number of clusters in the dataset.
- Then is stores the binary matrix (`mlb`), the `kmeans` object and the `clusters` object in a pickel file.
- For more details, check the comments in the code.

## `save_llama_neurons` function:
- First, it loops over every cluster and gets the sentences that belong to this cluster.
- Then calculate the activations of each sentence.
- Then for each layer, find the core neurons.
- In summary, it finds for each cluster, the core neurons in every layer.
- For more details, check the comments in the code.
