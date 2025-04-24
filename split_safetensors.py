import os
import sys
import torch
import json
from pathlib import Path
from collections import OrderedDict
from safetensors.torch import load_file, save_file  # Ensure you have safetensors package installed


def split_state_dict(state_dict, max_size_mb: int, base_path, prefix="pytorch_model-split", start_index=0):
    max_size_bytes = max_size_mb * 1024 * 1024
    current_split = OrderedDict()
    current_size = 0
    part_num = start_index
    index_mapping = {}

    for key, value in state_dict.items():
        tensor_bytes = value.element_size() * value.nelement()

        # If adding this tensor exceeds limit, save current split
        if current_size + tensor_bytes > max_size_bytes and current_split:
            split_file = base_path / f"{prefix}-{part_num:05d}.safetensors"
            save_file(current_split, split_file)  # Using safetensors to save
            for k in current_split:
                index_mapping[k] = split_file.name
            print(f"Saved {split_file} ({current_size / 1024 / 1024:.2f} MB)")
            part_num += 1
            current_split = OrderedDict()
            current_size = 0

        current_split[key] = value
        current_size += tensor_bytes

    # Save the final chunk
    if current_split:
        split_file = base_path / f"{prefix}-{part_num:05d}.safetensors"
        save_file(current_split, split_file)
        for k in current_split:
            index_mapping[k] = split_file.name
        print(f"Saved {split_file} ({current_size / 1024 / 1024:.2f} MB)")
        part_num += 1

    return part_num, index_mapping


def update_index_file(index_path, new_mapping, new_safetensors_files):
    with open(index_path, "r") as f:
        index_data = json.load(f)

    index_data["weight_map"] = new_mapping
    index_data["metadata"]["total_size"] = sum(Path(index_path.parent / fname).stat().st_size for fname in new_safetensors_files)

    # Save updated index file
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"Updated {index_path} with new weight map.")


def main(model_dir, max_size_mb):
    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("*.safetensors"))
    index_file = model_dir / "model.safetensors.index.json"

    full_mapping = {}
    start_index = 0

    for file in files:
        print(f"Loading {file.name}...")
        state_dict = load_file(file)  # Load safetensors file
        start_index, part_mapping = split_state_dict(state_dict, max_size_mb, model_dir, start_index=start_index)
        full_mapping.update(part_mapping)

    new_safetensors_files = sorted(set(full_mapping.values()))
    update_index_file(index_file, full_mapping, new_safetensors_files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("First argument needs to be the path to the model to split")
    
    model_dir = sys.argv[1]
    max_size_mb = 150
    main(model_dir, max_size_mb)
