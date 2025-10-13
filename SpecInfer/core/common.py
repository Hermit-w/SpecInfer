import logging
import time
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING
import os

import torch
import transformers

if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType

logger = logging.getLogger(__name__)
rank = int(os.environ.get("RANK", 0))

@dataclass
class InputForCasualLm:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[transformers.Cache]

    def to(
        self,
        device: "DeviceLikeType"
    ) -> "InputForCasualLm":
        _input_ids = self.input_ids.to(device)
        _attention_mask = self.attention_mask.to(device)
        # Try to move past_key_values to device if present. Some Cache
        # implementations provide a .to(device) method; if not, keep as-is.
        _past = None
        if self.past_key_values is not None:
            to_fn = getattr(self.past_key_values, "to", None)
            if callable(to_fn):
                try:
                    _past = to_fn(device)
                except Exception:
                    # Fallback: leave past_key_values as-is (may require
                    # caller to handle device placement).
                    _past = self.past_key_values
            else:
                _past = self.past_key_values

        return InputForCasualLm(
            _input_ids,
            _attention_mask,
            _past,
        )

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
    
    def __repr__(self):
        return f"""InputForCasualLm:
                \tinput_ids: {self.input_ids.shape}
                \tattention_mask: {self.attention_mask.shape}
                \tpast_key_values: {self.past_key_values.get_seq_length() if self.past_key_values is not None else 'none'}
                """

@dataclass
class OutputForCasualLm:
    generated_length: int
    output_ids: Optional[torch.Tensor]
    output_logits: torch.Tensor
    output_distribution: torch.Tensor
    past_key_values: Optional[transformers.Cache]

    def __repr__(self):
        return f"""OutputForCasualLm: 
                \tgenerated_length: {self.generated_length}
                \toutput_ids: {self.output_ids.shape if self.output_ids is not None else 'none'}
                \toutput_logits: {self.output_logits.shape}
                \toutput_distribution: {self.output_distribution.shape}
                \tpast_key_values: {self.past_key_values.get_seq_length() if self.past_key_values is not None else 'none'}
                """


def target_sample_from_distribution(
    target_distribution: torch.Tensor,
    draft_distribution: torch.Tensor
) -> torch.Tensor:
    diff_distribution = target_distribution - draft_distribution
    diff_distribution = torch.max(
        diff_distribution,
        torch.zeros_like(diff_distribution)
    )
    if (diff_distribution.sum(dim=-1, keepdim=True) == 0).any():
        diff_distribution = torch.where(
            diff_distribution == 0,
            diff_distribution + 1e-10,
            diff_distribution
            )
        logger.warning("Distribution contains zero values")
    diff_distribution = diff_distribution / \
        diff_distribution.sum(dim=-1, keepdim=True)
    logger.debug(diff_distribution.shape)
    sample_tokens = torch.multinomial(diff_distribution, num_samples=1)
    logger.debug(sample_tokens.shape)
    return sample_tokens


def speculative_sample(
    proposer_output: OutputForCasualLm,
    verifier_output: OutputForCasualLm,
) -> torch.Tensor:
    assert proposer_output.output_ids is not None, "The proposer's output contains None ids"
    length = proposer_output.generated_length
    bs = verifier_output.output_distribution.shape[0]
    verifier_distribution = verifier_output.output_distribution[:, -length:, :]
    proposer_distribution = proposer_output.output_distribution
    accept_tokens: list[torch.Tensor] = []
    # Pre-generate synchronized random numbers for accept/reject decisions.
    # If torch.distributed is initialized, create on rank 0 and broadcast to all ranks
    # to ensure consistent randomness across processes.
    use_distributed = False
    try:
        use_distributed = torch.distributed.is_initialized()
    except Exception:
        use_distributed = False

    # sample_ratio has shape [batch_size]
    # We'll prepare a random tensor of shape [length, batch_size]
    device = proposer_output.output_ids.device
    if use_distributed:
        # create on CPU to simplify cross-device broadcasting, then move to device when used
        rand_tensor = torch.empty(length, dtype=proposer_distribution.dtype, device=device)
        if rank == 0:
            # fill random values on rank 0
            rand_tensor.uniform_(0.0, 1.0)
        # broadcast from rank 0 to all processes
        torch.distributed.broadcast(rand_tensor, src=0)
    else:
        rand_tensor = torch.rand(length, device=device, dtype=proposer_distribution.dtype)

    for i in range(length):
        # Accept-Reject step
        logger.debug(f"Speculative sampling step {i}")
        logger.debug(f"Proposer token id: {proposer_output.output_ids[:, i]}")
        logger.debug(f"Proposer distribution: {proposer_distribution[:, i, proposer_output.output_ids[:, i]]}")
        logger.debug(f"Verifier distribution: {verifier_distribution[:, i, proposer_output.output_ids[:, i]]}")
        sample_ratio = verifier_distribution[:, i, proposer_output.output_ids[0, i]] / proposer_distribution[:, i, proposer_output.output_ids[0, i]]
        sample_ratio = torch.min(
            sample_ratio,
            torch.ones_like(sample_ratio)
        )
        rs = rand_tensor[i]
        logger.debug(f"Sample ratio: {sample_ratio}")
        logger.debug(f"Random value: {rs}")
        if rs < sample_ratio:
            logger.debug(f"Accept token {proposer_output.output_ids[:, i]}")
            accept_tokens.append(proposer_output.output_ids[:, i])
        else:
            if use_distributed:
                sample_tokens = torch.empty((bs, 1), dtype=torch.int64, device=device)
                if rank == 0:
                    sample_tokens = target_sample_from_distribution(
                        verifier_distribution[:, i, :],
                        proposer_distribution[:, i, :]
                    )
                torch.distributed.broadcast(sample_tokens, src=0)
            else:
                sample_tokens = target_sample_from_distribution(
                    verifier_distribution[:, i, :],
                    proposer_distribution[:, i, :]
                )
            accept_tokens.append(sample_tokens)
            logger.debug(f"Reject token {proposer_output.output_ids[:, i]}")
            logger.debug(f"Sample token {sample_tokens}")
            break
    else:
        logger.debug("All tokens accepted")
        if use_distributed:
            sample_tokens = torch.empty((bs, 1), dtype=torch.int64, device=device)
            if rank == 0:
                sample_tokens = torch.multinomial(verifier_distribution[:, -1, :], num_samples=1).squeeze(-1)
            torch.distributed.broadcast(sample_tokens, src=0)
        else:
            sample_tokens = torch.multinomial(verifier_distribution[:, -1, :], num_samples=1).squeeze(-1)
        accept_tokens.append(sample_tokens)
        logger.debug(f"Sample token {sample_tokens}")
    return torch.cat(accept_tokens, dim=-1)


def synchronize_time() -> float:
    torch.cuda.synchronize()
    return time.time()
