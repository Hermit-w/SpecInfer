import json
import logging
import os
from typing import TYPE_CHECKING

from safetensors import safe_open
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               SequenceParallel)
from transformers import AutoConfig
from transformers import Qwen3MoeForCausalLM
from torch.distributed.tensor.parallel import ParallelStyle
from transformers import PretrainedConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DECODER_ONLY_MODELS = [
    Qwen3MoeForCausalLM,
]

ALL_SUPPORTED_MODELS = [
    cls.__name__ for cls in DECODER_ONLY_MODELS
]


def check_support_model(
    model_config: PretrainedConfig
) -> bool:
    model_architecture = model_config.architectures[0] if model_config.architectures is not None else ""
    if model_architecture not in ALL_SUPPORTED_MODELS:
        logger.error(f"Model {model_architecture} is not supported yet.")
        logger.error(f"Supported models are listed here: {ALL_SUPPORTED_MODELS}")
    return True

def check_tensor_parallel_legality(
    model_config: PretrainedConfig,
    tp_size: int,
) -> bool:
    num_kv_heads = model_config.num_key_value_heads
    num_q_heads = model_config.num_attention_heads
    if num_q_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
        logger.error(f"The number of attention heads can not split by tp={tp_size}")
        return False
    
    return True


def read_weight_names(
    model_name_or_path: str,
) -> list[str]:
    weight_names: list[str] = []
    index_file = os.path.join(model_name_or_path, "model.safetensors.index.json")
    if os.path.isfile(index_file):
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        weight_names = list(weight_map.keys())
    else:
        model_file = os.path.join(model_name_or_path, "model.safetensors")
        with safe_open(model_file, framework='pt', device='cpu') as f:
            weight_names = f.keys()
    return weight_names


def get_tensor_parallel_plan(
    model_name_or_path: str,
    model_config: PretrainedConfig,
) -> dict[str, ParallelStyle]:
    model_architecture = model_config.architectures[0] if model_config.architectures is not None else ""
    if model_architecture in [m.__name__ for m in DECODER_ONLY_MODELS]:
        tp_plan: dict[str, ParallelStyle] = {}
        param_names = read_weight_names(model_name_or_path)
        for name in param_names:
            # layernorm
            if "norm" in name:
                tp_plan[name] = SequenceParallel()
            # logits_head
            elif "lm_head" in name:
                tp_plan[name] = RowwiseParallel()
            # embedding layer
            elif "embed_tokens" in name:
                tp_plan[name] = RowwiseParallel()
            # self-attention part
            elif "k_proj" in name:
                tp_plan[name] = ColwiseParallel()
            elif "q_proj" in name:
                tp_plan[name] = ColwiseParallel()
            elif "v_proj" in name:
                tp_plan[name] = ColwiseParallel()
            elif "o_proj" in name:
                tp_plan[name] = RowwiseParallel()
            # mlp part
            elif "up_proj" in name:
                tp_plan[name] = ColwiseParallel()
            elif "down_proj" in name:
                tp_plan[name] = RowwiseParallel()
            elif "gate_proj" in name:
                tp_plan[name] = ColwiseParallel()
        return tp_plan
    else:
        raise ValueError(f"Model architecture {model_architecture} is not supported for tensor parallelism.")


if __name__ == "__main__":
    model_path = "/export/home/lanliwei.1/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
    config = AutoConfig.from_pretrained(model_path)
    print(config)
    # get_tensor_parallel_plan(config)