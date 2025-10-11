import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from SpecInfer.core.common import InputForCasualLm, OutputForCasualLm, synchronize_time

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

logger = logging.getLogger(__name__)

rank = int(os.environ.get("RANK", 0))


class Proposer(ABC):
    def __init__(
        self,
        benchmark_time: bool
    ):
        self.propose_time_list: list[float] = []
        self.adjust_time_list: list[float] = []

        self.benchmark_time: bool = benchmark_time

    def propose(
        self,
        inputs: Union[InputForCasualLm, list[InputForCasualLm]],
        n: int,
        sample_method: Callable[[torch.Tensor], torch.Tensor],
    ) -> Union[OutputForCasualLm, list[OutputForCasualLm]]:
        if self.benchmark_time:
            start = synchronize_time()

        ret = self.propose_impl(inputs, n, sample_method)

        if self.benchmark_time:
            end = synchronize_time()
            self.propose_time_list.append(start - end)

        return ret

    @abstractmethod
    def propose_impl(
        self,
        inputs: Union[InputForCasualLm, list[InputForCasualLm]],
        n: int,
        sample_method: Callable[[torch.Tensor], torch.Tensor]
    ) -> Union[OutputForCasualLm, list[OutputForCasualLm]]:
        raise NotImplementedError

    def adjust_input(
        self,
        accept_token_ids: torch.Tensor,
        proposer_input: InputForCasualLm,
        proposer_output: OutputForCasualLm,
    ) -> InputForCasualLm:
        if self.benchmark_time:
            start = synchronize_time()

        ret = self.adjust_input_impl(
            accept_token_ids,
            proposer_input,
            proposer_output
        )

        if self.benchmark_time:
            end = synchronize_time()
            self.adjust_time_list.append(start - end)

        return ret

    @abstractmethod
    def adjust_input_impl(
        self,
        accept_token_ids: torch.Tensor,
        proposer_input: InputForCasualLm,
        proposer_output: OutputForCasualLm,
    ) -> InputForCasualLm:
        raise NotImplementedError

    def print_time(self):
        if self.benchmark_time:
            logger.info(f"[Proposer] prompt phase: {self.propose_times[0]},",
                        f" decode phase: {np.median(self.propose_times[1:])},",
                        f" adjust time: {self.adjust_time}")


class RandomProposer(Proposer):
    def __init__(self, benchmark_time: bool):
        super().__init__(benchmark_time)

    def propose_impl(
        self,
        input,
        n: int,
        sample_method: Callable[[torch.Tensor], torch.Tensor],
    ) -> OutputForCasualLm:
        raise NotImplementedError
        return OutputForCasualLm(1, torch.randint(0, 32000, (1, 16), device='cuda'), None)

    def adjust_input_impl(
        self,
        accept_token_ids: torch.Tensor,
        proposer_input: InputForCasualLm,
        proposer_output: OutputForCasualLm,
    ) -> InputForCasualLm:
        raise NotImplementedError


class ModelWithCacheProposer(Proposer):
    def __init__(
        self,
        model: "_BaseModelWithGenerate",
        tokenizer: "PreTrainedTokenizerBase",
        benchmark_time: bool = False,
    ):
        super().__init__(benchmark_time)
        self.model: "_BaseModelWithGenerate" = model
        self.tokenizer: "PreTrainedTokenizerBase" = tokenizer

    def propose_impl(
        self,
        inputs,
        n,
        sample_method
    ) -> Union[OutputForCasualLm, list[OutputForCasualLm]]:
        if isinstance(inputs, list):
            if len(inputs) > 1:
                raise NotImplementedError("Only support bs=1 in current version")
            else:
                assert len(inputs) == 1, f"Expected bs=1, but get {len(inputs)}"
                inputs = inputs[0]

        use_distributed = False
        try:
            use_distributed = torch.distributed.is_initialized()
        except Exception:
            use_distributed = False

        propose_tokens_list: list = []
        propose_logits_list: list = []
        propose_distributions_list: list = []
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        past_key_values = inputs.past_key_values
        generated_len = n
        for i in range(n):
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_logits_ = outputs.logits
            past_key_values = outputs.past_key_values
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                ],
                dim=-1
            )

            if next_token_logits_ is None:
                logger.error(
                    f"{self.__class__.__name__}: "
                    "There is None in Model's outputs"
                    )
                raise ValueError

            next_token_logits = next_token_logits_[:, -1, :]
            distribution = sample_method(next_token_logits)

            if use_distributed:
                next_token_id = torch.empty((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
                if rank == 0:
                    next_token_id = torch.multinomial(distribution, num_samples=1)
                torch.distributed.broadcast(next_token_id, src=0)
            else:
                next_token_id = torch.multinomial(distribution, num_samples=1)

            propose_logits_list.append(next_token_logits.unsqueeze(1))
            propose_distributions_list.append(distribution.unsqueeze(1))
            propose_tokens_list.append(next_token_id)

            # Avoid calling .item() on CUDA tensors to prevent implicit
            # host-device synchronization which can interfere with
            # distributed runs. Use tensor comparison instead.
            eos_mask = (next_token_id == self.tokenizer.eos_token_id)
            if eos_mask.any():
                generated_len = i + 1
                logger.info(f"Stop at step {generated_len} because of eos")
                break

            input_ids = next_token_id
        propose_tokens = torch.cat(propose_tokens_list, dim=-1)
        propose_logits = torch.cat(propose_logits_list, dim=1)
        propose_distributions = torch.cat(propose_distributions_list, dim=1)

        return OutputForCasualLm(
            generated_len,
            propose_tokens,
            propose_logits,
            propose_distributions,
            past_key_values
        )

    def adjust_input_impl(
        self,
        accept_token_ids,
        proposer_input,
        proposer_output
    ):
        proposer_input_ids = accept_token_ids.tile(
            proposer_input.input_ids.shape[0], 1
        )
        proposer_attn_masks = torch.cat(
            [
                proposer_input.attention_mask,
                torch.ones_like(proposer_input_ids, dtype=torch.long)
            ],
            dim=-1
        )
        past_key_values = proposer_output.past_key_values

        total_generated_length = past_key_values.get_seq_length()
        past_key_values.crop(
            total_generated_length - proposer_output.generated_length
        )
        return InputForCasualLm(
            proposer_input_ids,
            proposer_attn_masks,
            past_key_values
        )
