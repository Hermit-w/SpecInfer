import logging
import os
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

from .common import InputForCasualLm, OutputForCasualLm, synchronize_time

logger = logging.getLogger(__name__)

rank = int(os.environ.get("RANK", 0))


class Verifier:
    def __init__(
        self,
        model: "_BaseModelWithGenerate",
        tokenizer: "PreTrainedTokenizerBase",
        benchmark_time: bool = False
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        self.verify_time: list[float] = []
        self.prepare_input_time: float = 0.0
        self.adjust_input_time: float = 0.0
        self.benchmark_time: bool = benchmark_time

    def verify(
        self,
        verifier_input: InputForCasualLm,
        proposer_output: OutputForCasualLm,
        sample_method: Callable[[torch.Tensor], torch.Tensor],
    ) -> OutputForCasualLm:
        if self.benchmark_time:
            start = synchronize_time()
        assert proposer_output.output_ids is not None, "The proposer's output contains None ids"

        if verifier_input.past_key_values is None:
            # concatenate proposed inputs with prompts
            input_ids = torch.cat(
                [verifier_input.input_ids, proposer_output.output_ids],
                dim=-1
            )
            # concatenate prompt masks with proposed token masks
            attention_mask = torch.cat(
                [
                    verifier_input.attention_mask,
                    torch.ones_like(
                        proposer_output.output_ids,
                        dtype=torch.long,
                        device=verifier_input.attention_mask.device
                    )
                ],
                dim=-1
            )
            # prompt phase, we don't have kv cache (past_key_values)
            past_key_values = None
        else:
            input_ids = torch.cat(
                [verifier_input.input_ids, proposer_output.output_ids],
                dim=-1
            )
            attention_mask = torch.cat(
                [
                    verifier_input.attention_mask,
                    torch.ones_like(
                        proposer_output.output_ids,
                        dtype=torch.long,
                        device=verifier_input.attention_mask.device
                    )
                ],
                dim=-1
            )

            past_key_values = verifier_input.past_key_values

        if self.benchmark_time:
            end = synchronize_time()
            self.prepare_input_time += (end - start)
            start = end

        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        token_logits = outputs.logits
        propose_len = proposer_output.generated_length
        if token_logits is None:
            logger.error("The logits returned by model is None.")
            raise ValueError("The logits in the output is None.")
        logits = token_logits[:, -propose_len:, :]
        distribution = sample_method(logits)

        if self.benchmark_time:
            self.verify_time.append(synchronize_time() - start)
        # output logits/distribution has shape [# of proposed tokens, vocab_size]
        return OutputForCasualLm(
            propose_len,
            None,
            logits,
            distribution,
            outputs.past_key_values
        )

    def adjust_input(
        self,
        accept_token_ids: torch.Tensor,
        verifier_output: OutputForCasualLm
    ) -> InputForCasualLm:
        if self.benchmark_time:
            start = synchronize_time()

        bs = accept_token_ids.shape[0]
        n_matches = accept_token_ids.shape[1] - 1
        logger.debug(f"Accept {n_matches} tokens")

        verifier_key_values = verifier_output.past_key_values

        verifier_valid_len = verifier_key_values.get_seq_length() - verifier_output.generated_length + n_matches \
            if verifier_key_values is not None else 0

        logger.debug(f"Original kv length: {verifier_key_values.get_seq_length() if verifier_key_values is not None else 0}")
        verifier_key_values.crop(verifier_valid_len) if verifier_key_values is not None else None
        logger.debug(f"Crop to length: {verifier_key_values.get_seq_length() if verifier_key_values is not None else 0}")
        verifier_attn_masks = torch.ones((bs, verifier_valid_len + 1), dtype=torch.int64, device=self.model.device)

        if self.benchmark_time:
            self.adjust_input_time += synchronize_time() - start

        return InputForCasualLm(
            accept_token_ids[:, -1].unsqueeze(1),
            verifier_attn_masks,
            verifier_key_values
        )

    def print_time(self):
        if self.benchmark_time:
            print(f"[Verifier] prompt phase: {self.verify_time[0]}, "
                  f"decode phase: {np.median(self.verify_time[1:])}, ",
                  f"adjust time: {self.adjust_input_time}, ",
                  f"prepare input time: {self.prepare_input_time}")