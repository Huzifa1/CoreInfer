from datasets import load_dataset
from huggingface_hub import snapshot_download
from pathlib import Path
def download_model_data(model_name, checkpoint_path, data_name, data_config, datasave_path, token = None):
    dataset = load_dataset(data_name, data_config)
    dataset.save_to_disk(datasave_path)
    
    local_dir_use_symlinks = False
    snapshot_download(
        repo_id=model_name,
        local_dir=checkpoint_path,
        local_dir_use_symlinks=local_dir_use_symlinks,
        token=token
    )
    print("data and model has been downloaded!")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="facebook/opt-6.7b", help='Model Name')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("./models/opt-6.7b"), help='Model checkpoint path.')
    parser.add_argument('--data_name', type=str, default="truthfulqa/truthful_qa", help='Data Name')
    parser.add_argument('--data_config', type=str, default="generation", help='Data Config')
    parser.add_argument('--datasave_path', type=str, default="./dataset/trurthul_qa", help='Data Path')
    parser.add_argument('--token', type=str, default=None, help='Model Token')

    args = parser.parse_args()

    download_model_data(args.model_name, args.checkpoint_path, args.data_name, args.data_config, args.datasave_path, args.token)