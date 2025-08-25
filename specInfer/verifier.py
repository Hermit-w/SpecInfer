import numpy as np
import torch
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from common import InputForCasualLm, OutputForCasualLm, synchronize_time
import logging

logger = logging.getLogger(__name__)

class Verifier:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        benchmark_time: bool = False
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        self.verify_times: list[float] = []
        self.prepare_input_time: float = 0.0
        self.adjust_input_time: float = 0.0
        self.benchmark_time: bool = benchmark_time

    def compare_tokens(
        self,
        proposed_output: OutputForCasualLm,
        verified_output: OutputForCasualLm
    ) -> torch.Tensor:
        if not proposed_output.output_ids.shape == \
           verified_output.output_ids[:, :-1].shape:
            logger.error(
                f"{proposed_output.output_ids.shape}, \
                  {verified_output.output_ids[:, :-1].shape}"
            )
            raise ValueError("Not compatiable shape")

        # a = [[1, 2, 3]], b = [[1, 2, 4]]
        # ~(a == b): [[0, 0, 1]]
        # after cumsum: [[0, 0, 1]]
        # after < 1: [[1, 1, 0]]
        n_matches = ((~(proposed_output.output_ids ==
                     verified_output.output_ids[:, :-1])).cumsum(dim=-1) < 1).sum()
        return verified_output.output_ids[:, :n_matches + 1], -1, -1

    def verify(
        self,
        inputs: InputForCasualLm,
        propose_len: int,
        logits_to_scores
    ) -> OutputForCasualLm:
        if self.benchmark_time:
            start = synchronize_time()

        outputs: CausalLMOutputWithPast = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=inputs.past_key_values,
            use_cache=True
        )

        if outputs.logits is None:
            logger.error("Get none logits from model!")
            raise ValueError
        modified_logits = outputs.logits
        generated_len = propose_len + 1
        logits = modified_logits[:, -generated_len:, :]
        scores = logits_to_scores(logits)

        if self.benchmark_time:
            self.verify_times.append(synchronize_time() - start)

        return OutputForCasualLm(
            generated_len,
            None,
            logits,
            scores,
            outputs.past_key_values
        )

    def prepare_input(
        self,
        proposer_output: OutputForCasualLm,
        verifier_input: InputForCasualLm
    ) -> InputForCasualLm:
        """
        Concatenate the proposer_output and verifier_input
        """
        if self.benchmark_time:
            start = synchronize_time()

        if proposer_output.output_ids is None:
            logger.error("Get none output_ids!")
            raise ValueError

        if verifier_input.past_key_values is None:
            # concatenate proposed inputs with prompts
            input_ids = torch.cat(
                [verifier_input.input_ids, proposer_output.output_ids], dim=-1)
            # concatenate prompt masks with proposed token masks
            attention_mask = torch.cat([verifier_input.attention_mask,
                                        torch.ones_like(proposer_output.output_ids,
                                                        dtype=torch.long, device="cuda")], dim=-1)
            # prompt phase, we don't have kv cache (past_key_values)
            past_key_values = None
        else:
            input_ids = torch.cat([verifier_input.input_ids.unsqueeze(
                0), proposer_output.output_ids], dim=-1)
            attention_mask = torch.cat([verifier_input.attention_mask,
                                        torch.ones_like(proposer_output.output_ids,
                                                        dtype=torch.long, device="cuda")], dim=-1)

            past_key_values = verifier_input.past_key_values

        if self.benchmark_time:
            self.prepare_input_time += synchronize_time() - start

        return InputForCasualLm(input_ids, attention_mask, past_key_values)

    def adjust_input(
        self,
        accept_token_ids: torch.Tensor,
        verifier_input: InputForCasualLm,
        verifier_output: OutputForCasualLm
    ) -> InputForCasualLm:
        if self.benchmark_time:
            start = synchronize_time()

        n_matches = accept_token_ids.shape[1]
        if str(self.model.__class__.__name__) in ["GPTBigCodeForCausalLM"]:
            verifier_generated_len = verifier_output.past_key_values[0].shape[-2] - (
                verifier_output.generated_len - 1) + n_matches
            verifier_key_values = crop_mqa_past_key_values(
                verifier_output.past_key_values, verifier_generated_len - 1)
        else:
            verifier_generated_len = verifier_output.past_key_values[0][0].shape[2] - (
                verifier_output.generated_len - 1) + n_matches

            verifier_key_values = crop_past_key_values(
                verifier_output.past_key_values, verifier_generated_len - 1)

            verifier_attn_masks = verifier_input.attention_mask[:,
                                                                :verifier_generated_len]
            if verifier_attn_masks.shape[1] < verifier_generated_len:
                verifier_attn_masks = torch.cat([verifier_attn_masks,
                                                torch.ones(verifier_attn_masks.shape[0], 1, dtype=torch.long, device="cuda")], dim=-1)

        if self.benchmark_time:
            self.adjust_input_time += synchronize_time() - start
        return InputForCasualLm(
            accept_token_ids[:, -1],
            verifier_attn_masks,
            verifier_key_values
        )

    def print_time(self):
        if self.benchmark_time:
            print(f"[Verifier] prompt phase: {self.verify_times[0]}, "
                  f"decode phase: {np.median(self.verify_times[1:])}, ",
                  f"adjust time: {self.adjust_input_time}, ",
                  f"prepare input time: {self.prepare_input_time}")