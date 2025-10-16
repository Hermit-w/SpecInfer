import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import numpy as np

from .common import (InputForCasualLm, OutputForCasualLm, speculative_sample,
                     synchronize_time)
from .proposer import ModelWithCacheProposer
from .utils import get_softmax_func, decode
from .verifier import Verifier

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

logger = logging.getLogger(__name__)


@dataclass
class GeneratorOutput:
    output: str
    generated_ids: torch.Tensor
    correct_tokens: torch.Tensor
    propose_steps: int
    sample_steps: int
    alpha_sum: list[float]
    wrong_token_ids: list[int]

    def summary(self) -> str:
        return (
            f"The output is: {self.output}.\n"
            f"The target model forward {self.propose_steps} times.\n"
            f"We accept {self.correct_tokens.shape[1]} tokens.\n"
            f"The expected accept rate is {sum(self.alpha_sum)/self.sample_steps}."
        )


class SpecGenerator:
    def __init__(
        self,
        draft_model: "_BaseModelWithGenerate",
        target_model: "_BaseModelWithGenerate",
        tokenizer: "PreTrainedTokenizerBase",
        benchmark_time: bool = False,
    ):
        self.draft_model: "_BaseModelWithGenerate" = draft_model
        self.target_model: "_BaseModelWithGenerate" = target_model
        self.tokenizer: "PreTrainedTokenizerBase" = tokenizer

        # metrics
        self.benchmark_time: bool = benchmark_time
        self.generation_time: list[float] = []

        self.proposer = ModelWithCacheProposer(
            draft_model,
            tokenizer,
            benchmark_time,
        )
        self.verifier = Verifier(
            target_model,
            tokenizer,
            benchmark_time
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_propose_num: int,
        max_tokens: int,
        temperature: float = 0.01,
    ) -> GeneratorOutput:
        self.target_model.eval()
        self.draft_model.eval()

        sample_method = get_softmax_func(temperature)

        generated_token_cnt = 0
        generated_tokens_list = []
        wrong_token_ids = []

        proposer_input = InputForCasualLm(input_ids, attention_mask, None)
        verifier_input = copy.deepcopy(proposer_input)
        correct_tokens_list = []
        propose_steps = 0
        alpha: list[float] = []
        sample_steps = 0
        while True:
            if self.benchmark_time:
                start = synchronize_time()
            # propose n tokens, proposer always propose the token with highest probability
            proposer_output = self.proposer.propose(
                proposer_input,
                max_propose_num,
                sample_method
            )

            propose_steps += 1

            # forward n tokens on the model in the a single run
            verifier_output = self.verifier.verify(
                verifier_input,
                proposer_output,
                sample_method
            )

            # compare selected tokens
            accept_token_ids, cur_alpha, cur_sample_steps = self._sample_tokens(
                proposer_output, verifier_output
            )
            logger.debug(f"Accept_token: {decode(self.tokenizer, accept_token_ids)}")
            alpha.extend(cur_alpha)
            sample_steps += cur_sample_steps
            # logger.log("acc_tokens", accept_token_ids)
            generated_tokens_list.append(accept_token_ids)
            generated_token_cnt += accept_token_ids.shape[1]
            wrong_token_ids.append(generated_token_cnt - 1)

            correct_tokens_list.append(accept_token_ids[:, :-1])

            # adjust the proposer/verifier input, discard unnecessary kv cache
            proposer_input = self.proposer.adjust_input(
                accept_token_ids, proposer_input, proposer_output)
            verifier_input = self.verifier.adjust_input(
                accept_token_ids, verifier_output)

            if self.benchmark_time:
                self.generation_time.append(synchronize_time() - start)

            if generated_token_cnt >= max_tokens \
               or self.tokenizer.eos_token_id in accept_token_ids:
                break

        # self.proposer.print_time()
        # self.verifier.print_time()
        # self.print_time()
        generated_tokens = torch.cat(generated_tokens_list, dim=-1)
        correct_tokens = torch.cat(correct_tokens_list, dim=-1)

        return GeneratorOutput(
            "".join(self.tokenizer.batch_decode(generated_tokens)),
            generated_tokens,
            correct_tokens,
            propose_steps,
            sample_steps,
            alpha,
            wrong_token_ids,
        )

    def _sample_tokens(
        self,
        proposer_output: OutputForCasualLm,
        verifier_output: OutputForCasualLm,
    ) -> tuple[torch.Tensor, list[float], int]:
        return speculative_sample(proposer_output, verifier_output)

    def summary(self) -> str:
        if self.benchmark_time:
            return (
                f"Generator:\n"
                f"\tTotal time(prompt phase + decode phase): {np.sum(self.generation_time)}s.\n"
                f"\tGeneration time(prompt phase): {self.generation_time[0]}s.\n"
                f"\tGeneration time(decode phase): {np.median(self.generation_time[1:])}s.\n"
                f"\tMax allocated memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB."
            )
        else:
            return ""

    def print(self):
        if self.benchmark_time:
            logger.info("\n" + self.summary())
