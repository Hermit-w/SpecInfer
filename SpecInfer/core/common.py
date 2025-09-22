import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch
import transformers

logger = logging.getLogger(__name__)


@dataclass
class InputForCasualLm:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[transformers.Cache]

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> "InputForCasualLm":
        inputs = tokenizer(prompt, return_tensors="pt")
        return InputForCasualLm(
            inputs['input_ids'],
            inputs['attention_mask'],
            None,
        )

@dataclass
class OutputForCasualLm:
    generated_length: int
    output_ids: torch.Tensor
    output_logits: torch.Tensor
    output_distribution: torch.Tensor
    past_key_values: Optional[transformers.Cache]


def target_sample_from_distribution(
        target_distribution: torch.Tensor,
        draft_distribution: torch.Tensor
        ) -> torch.Tensor:
    diff_distribution = target_distribution - draft_distribution
    diff_distribution = torch.max(diff_distribution,
                                  torch.zeros_like(diff_distribution))
    if (diff_distribution.sum(dim=-1, keepdim=True) == 0).any():
        diff_distribution = torch.where(
            diff_distribution == 0,
            diff_distribution + 1e-10,
            diff_distribution
            )
        logger.warning("Distribution contains zero values")
    diff_distribution = diff_distribution / \
        diff_distribution.sum(dim=-1, keepdim=True)
    return torch.multinomial(diff_distribution, num_samples=1).squeeze(-1)


def synchronize_time() -> float:
    torch.cuda.synchronize()
    return time.time()
