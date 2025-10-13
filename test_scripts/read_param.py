import json
import os
from typing import Optional

import yaml
from safetensors import safe_open

INDEX_FILE = "model.safetensors.index.json"


def read_param(model_name_or_path: str, output_file: Optional[str]):
    file_path = os.path.join(model_name_or_path, INDEX_FILE)
    all_shapes = {}
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            index_data = json.load(f)

        weight_map = index_data["weight_map"]
        shard_files = set(weight_map.values())

        for shard_file in shard_files:
            shard_path = os.path.join(model_name_or_path, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                tensors = f.keys()
                for name in tensors:
                    shape = f.get_tensor(name).shape
                    all_shapes[name] = list(shape)
    else:
        model_file = os.path.join(model_name_or_path, "model.safetensors")
        with safe_open(model_file, framework='pt', device='cpu') as f:
            tensors = f.keys()
            for name in tensors:
                shape = f.get_tensor(name).shape
                all_shapes[name] = list(shape)
                # print(all_shapes)
    if output_file is not None:
        yaml.safe_dump(all_shapes, open(output_file, "w"), sort_keys=True)


if __name__ == '__main__':
    prefix_name = "/export/home/lanliwei.1/models"
    model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    full_name = os.path.join(prefix_name, model_name)
    output_name = model_name.split("/")[1].lower() + ".yaml"
    read_param(full_name, output_name)
