import json
import random
import sys

def shuffle_neurons(input_file, output_file):
    # Read the JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Shuffle each list of neurons
    for neuron_list in data.get("neurons", []):
        random.shuffle(neuron_list)

    # Write the new JSON to output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python randomize_neurons.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]  # Input JSON file
    output_file = sys.argv[2]  # Output JSON file
    
    shuffle_neurons(input_file, output_file)
    print(f"Shuffled JSON written to {output_file}")
