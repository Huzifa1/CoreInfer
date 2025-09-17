import json

def load_and_normalize_activations(file_path, token_count):
    with open(f"statistics_files/{file_path}", "r") as file:
        lines = file.readlines()[2:]
        activations = []
        for line in lines:
            parts = line.split(":")[1].split(",")
            activations.append([float(x.strip()) / token_count for x in parts])
    return activations

datasets = {
    # "triviaqa": {"path": "triviaqa.statistics"},
    # "squadv2": {"path": "squadv2.statistics"},
    # "mlqa": {"path": "mlqa.statistics"},
    "piqa": {"path": "piqa.statistics"},
    # "wmt16-de-en": {"path": "wmt16-de-en.statistics"},
    # "wmt16-ro-en": {"path": "wmt16-ro-en.statistics"},
    # "wmt14-fr-en": {"path": "wmt14-fr-en.statistics"},
    "wmt14-en-fr": {"path": "wmt14-en-fr.statistics"},
    # "cnn_dailymail": {"path": "cnn_dailymail.statistics"},
    # "xsum": {"path": "xsum.statistics"},
    "samsum": {"path": "samsum.statistics"},
}

outputFileName = f"../neuron_files/opt-6.7b/model_neurons.json"

normalized_activations = {}
for name, meta in datasets.items():
    with open(f'statistics_files/{meta["path"]}', "r") as f:
        lines = f.readlines()
        num_tokens = int(lines[-1].split("Number of tokens: ")[1])
        print(num_tokens)
    normalized_activations[name] = load_and_normalize_activations(meta["path"], num_tokens)

num_layers = len(next(iter(normalized_activations.values()))) - 1
finalResult = []
for i in range(num_layers):
    combined = zip(*(normalized_activations[ds][i] for ds in datasets))
    averaged_layer = [sum(vals) / len(vals) * 100 for vals in combined]
    finalResult.append(averaged_layer)


finalSorted = []
for layer in finalResult:
    sorted_with_indices = sorted(enumerate(layer), key=lambda x: x[1], reverse=True)
    sorted_indices = [i for i, _ in sorted_with_indices]
    sorted_values = [v for _, v in sorted_with_indices]
    finalSorted.append(sorted_indices)

write_file = True
if write_file:
    with open(outputFileName, "w") as file:
        json.dump({"neurons": finalSorted}, file)





