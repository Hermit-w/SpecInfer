import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from common import InputForCasualLm, OutputForCasualLm, synchronize_time

logger = logging.getLogger(__name__)


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
        inputs: InputForCasualLm,
        n: int,
        logits_to_scores: Callable[[torch.Tensor], torch.Tensor],
    ) -> OutputForCasualLm:
        if self.benchmark_time:
            start = synchronize_time()

        ret = self.propose_impl(inputs, n, logits_to_scores)

        if self.benchmark_time:
            end = synchronize_time()
            self.propose_time_list.append(start - end)

        return ret

    @abstractmethod
    def propose_impl(
        self,
        inputs: InputForCasualLm,
        n: int,
        sample_method: Callable[[torch.Tensor], torch.Tensor]
    ) -> OutputForCasualLm:
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
        inputs: InputForCasualLm,
        n: int,
        logits_to_scores: Callable[[torch.Tensor], torch.Tensor],
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


class ModelProposer(Proposer):
    def __init__(
        self,
        benchmark_time: bool,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ):
        super().__init__(benchmark_time)
        self.model: transformers.PreTrainedModel = model
        self.tokenizer: transformers.PreTrainedTokenizerBase = tokenizer

    def propose_impl(
        self,
        inputs: InputForCasualLm,
        n: int,
        sample_method: Callable[[torch.Tensor], torch.Tensor]
    ) -> OutputForCasualLm:
        if inputs.input_ids.shape[0] > 1:
            raise NotImplementedError(
                "Not implement for batch_size > 1 in evaluation")
        raise NotImplementedError

    def adjust_input_impl(
        self,
        accept_token_ids,
        proposer_input,
        proposer_output
    ):
        raise NotImplementedError


class ModelWithCacheProposer(Proposer):
    def __init__(
        self,
        benchmark_time: bool,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ):
        super().__init__(benchmark_time)
        self.model: transformers.PreTrainedModel = model
        self.tokenizer: transformers.PreTrainedTokenizerBase = tokenizer

    def propose_impl(
        self,
        inputs: InputForCasualLm,
        n: int,
        logits_to_scores: Callable[[torch.Tensor], torch.Tensor],
    ) -> OutputForCasualLm:
        if inputs.input_ids.shape[0] > 1:
            raise NotImplementedError(
                "Not implement for batch_size > 1 in evaluation")
        propose_tokens_list: list[torch.Tensor] = []
        propose_logits_list: list[torch.Tensor] = []
        propose_distributions_list: list[torch.Tensor] = []
        input_ids = inputs.input_ids
        past_key_values = inputs.past_key_values
        generated_len: int = n
        for i in range(n):
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits_ = outputs.logits
            if next_token_logits_ is None:
                logger.error(
                    f"{self.__class__.__name__}: "
                    "There is None in Model's outputs"
                    )
                raise ValueError
            past_key_values = outputs.past_key_values
            next_token_logits = next_token_logits_[:, -1, :]
            distribution = logits_to_scores(next_token_logits)
            next_token_id = torch.multinomial(distribution, num_samples=1)

            propose_logits_list.append(next_token_logits.unsqueeze(1))  # recover the dim
            propose_distributions_list.append(distribution.unsqueeze(1))
            propose_tokens_list.append(next_token_id)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                generated_len = i + 1
                logger.info(f"Stop at {generated_len} because of eos")
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
            [proposer_input.attention_mask,
             torch.ones_like(proposer_input_ids, dtype=torch.long)], dim=-1
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
