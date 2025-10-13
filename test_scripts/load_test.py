import logging
import os
from typing import Optional, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer

from SpecInfer.model_loader.loader import ModelLoader
from SpecInfer.core.logger import setup_logger
import torch

setup_logger(level="debug")

model_path = "/export/home/lanliwei.1/models/Qwen/Qwen3-30B-A3B-Instruct-2507"


def get_memory_usage(device):
    allocated = torch.cuda.memory_allocated(device) / 1024**2  
    reserved = torch.cuda.memory_reserved(device) / 1024**2   
    return allocated, reserved


rank = 0
device = torch.device(f"cuda:{rank}")

# torch.distributed.init_process_group("nccl", device_id=device)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
).to(device)

print(model._tp_plan)

after_allocated, after_reserved = get_memory_usage(device)
print(f"allocate: {after_allocated:.2f} MB,reverse: {after_reserved:.2f} MB")

tokenizer = AutoTokenizer.from_pretrained(
    model_path
)

prompt = "Introduce yourself."

inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print("Yes Here")
print(inputs)
print(rank)

output = model(inputs)
print(output)

# if local_rank == 0:
#     print(output)