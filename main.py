import subprocess
import sys


def main(
    model_name_or_path: str,
    tp_size: int,
):
    cmd = [
        "torchrun",
        f"--nporc_per_node={tp_size}",
        "specInfer/specInfer.py"
        "--model", f"{model_name_or_path}"
        "-tp", f"{tp_size}"
    ]

    


if __name__ == '__main__':
    model_path = "/export/home/lanliwei.1/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
    tp_size = 2
    main(model_path, tp_size)