
outputFileName = "LLama_model_neurons_test.txt"

datasets = {
    "cooking": {"path": "cooking.statistics", "tokens": 184018},
    "fiqa": {"path": "fiqa.statistics", "tokens": 564018},
    "stqa": {"path": "stqa.statistics", "tokens": 433379},
    "truthful": {"path": "truthful.statistics", "tokens": 35256},
}

def load_and_normalize_activations(file_path, token_count):
    with open(file_path, "r") as file:
        lines = file.readlines()[2:]
        activations = []
        for line in lines:
            parts = line.split(":")[1].split(",")
            activations.append([float(x.strip()) / token_count for x in parts])
    return activations

normalized_activations = {}
for name, meta in datasets.items():
    normalized_activations[name] = load_and_normalize_activations(meta["path"], meta["tokens"])

num_layers = len(next(iter(normalized_activations.values())))
finalResult = []
for i in range(num_layers):
    combined = zip(*(normalized_activations[ds][i] for ds in datasets))
    averaged_layer = [sum(vals) / len(vals) * 100 for vals in combined]
    finalResult.append(averaged_layer)



counter = 0
threshhold = 100
indices = []
for layer in finalResult:
    indicesLayer = []
    i = 0
    for neuron in layer:
        if neuron >= threshhold:
            indicesLayer.append(i)
            counter += 1
        i += 1
    indices.append(indicesLayer)

counter2 = 0
for layer in indices:
    for neuron in layer:
        counter2 += 1
print(counter2)
#print(indices)


finalSorted = []
for layer in finalResult:
    sorted_with_indices = sorted(enumerate(layer), key=lambda x: x[1], reverse=True)
    sorted_indices = [i for i, _ in sorted_with_indices]
    sorted_values = [v for _, v in sorted_with_indices]
    finalSorted.append(sorted_indices)

write_file = True
if write_file:
    with open(outputFileName, "w") as file:
        for layer in finalSorted:
            file.write(str(layer) + "\n")
print("Number of neurons above threshold: ", counter)





