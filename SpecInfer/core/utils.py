from typing import Callable, TYPE_CHECKING

import torch

from .common import synchronize_time

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def get_softmax_func(temperature: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def sample_method(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor / temperature
        return torch.nn.functional.softmax(tensor, dim=-1)
    return sample_method


def benchmark_time(func: Callable) -> tuple[Callable, float]:
    start_time = synchronize_time()

    def inner_func(*args, **kwargs):
        return func(*args, **kwargs)
    end_time = synchronize_time()
    return inner_func, end_time - start_time


def decode(tokenizer: "PreTrainedTokenizerBase", ids: torch.Tensor) -> str:
    return "".join(tokenizer.batch_decode(ids))
