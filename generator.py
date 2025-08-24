import copy
import logging
from dataclasses import dataclass

import torch
import transformers

from common import InputForCasualLm, OutputForCasualLm, synchronize_time
from proposer import ModelProposer, ModelWithCacheProposer
from verifier import Verifier

logger = logging.getLogger(__name__)


@dataclass
class GeneratorOutput:
    output: list[str]
    generated_ids: torch.Tensor
    correct_tokens: torch.Tensor
    propose_steps: int
    sample_steps: int
    alpha_sum: float
    wrong_token_ids: list[int]


class SpecGenerator:
    def __init__(
        self,
        draft_model: transformers.PreTrainedModel,
        target_model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_propose_num: int,
        use_cache: bool,
        benchmark_time: bool = False,
    ):
        self.draft_model: transformers.PreTrainedModel = draft_model
        self.target_model: transformers.PreTrainedModel = target_model
        self.tokenizer: transformers.PreTrainedTokenizerBase = tokenizer
        self.max_propose_num: int = max_propose_num

        # metrics
        self.benchmark_time: bool = benchmark_time
        self.generation_time: list[float] = []

        self.proposer: ModelProposer | ModelWithCacheProposer
        if use_cache:
            self.proposer = ModelWithCacheProposer(
                benchmark_time,
                draft_model,
                tokenizer
            )
        else:
            self.proposer = ModelProposer(
                benchmark_time,
                draft_model,
                tokenizer
            )
        self.verifier = Verifier(
            target_model,
            tokenizer,
            benchmark_time
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        attention_mask: torch.Tensor,
        temperature: float = 0.01,
    ) -> GeneratorOutput:
        self.target_model.eval()
        self.draft_model.eval()

        def sample_method(logits: torch.Tensor):
            return torch.softmax(logits / temperature, dim=-1)

        generated_token_cnt = 0
        generated_tokens = None
        wrong_token_ids = []

        proposer_input = InputForCasualLm(input_ids, attention_mask, None)
        verifier_input = copy.deepcopy(proposer_input)
        correct_tokens = None
        propose_steps = 0
        alpha, sample_steps = 0, 0
        while True:
            start = synchronize_time()
            # propose n tokens, proposer always propose the token with highest probability
            proposer_output = self.proposer.propose(
                proposer_input,
                self.max_propose_num,
                sample_method
            )
            propose_steps += 1

            # prepare verifier input
            verifier_input = self.verifier.prepare_input(
                proposer_output,
                verifier_input
            )

            # forward n tokens on the model in the a single run
            verifier_output = self.verifier.verify(
                verifier_input,
                proposer_output.generated_length,
                sample_method)

            # compare selected tokens
            # accept_token_ids, cur_alpha, cur_sample_steps = self.compare_tokens(proposer_output, verifier_output)
            accept_token_ids, cur_alpha, cur_sample_steps = self.sample_tokens(
                proposer_output, verifier_output)
            alpha += cur_alpha
            sample_steps += cur_sample_steps
            # logger.log("acc_tokens", accept_token_ids)
            if generated_tokens is None:
                generated_tokens = accept_token_ids
            else:
                generated_tokens = torch.cat(
                    [generated_tokens, accept_token_ids], dim=-1)
            generated_token_cnt += accept_token_ids.shape[1]
            wrong_token_ids.append(generated_token_cnt - 1)

            if correct_tokens is None:
                correct_tokens = accept_token_ids[:, :-1]
            else:
                correct_tokens = torch.cat(
                    [correct_tokens, accept_token_ids[:, :-1]], dim=-1)

            # adjust the proposer/verifier input, discard unnecessary kv cache
            proposer_input = self.proposer.adjust_input(
                accept_token_ids, proposer_input, proposer_output)
            verifier_input = self.verifier.adjust_input(
                accept_token_ids, verifier_input, verifier_output)

            if self.benchmark_time:
                self.generation_time.append(synchronize_time() - start)

            if generated_token_cnt >= max_tokens \
               or self.tokenizer.eos_token_id in accept_token_ids:
                break

        self.proposer.print_time()
        self.verifier.print_time()
        self.print_time()
        return GeneratorOutput(
            self.tokenizer.batch_decode(generated_tokens),
            generated_tokens,
            correct_tokens,
            propose_steps,
            sample_steps,
            alpha,
            wrong_token_ids,
        )

    def print_time(self):
        if self.benchmark_time:
            logger.info(f"[Generator time]: {self.generation_time}")
            logger.info(
                f"[Max allocated memory]: \
                 {torch.cuda.max_memory_allocated() / 1024 / 1024} MB"
            )
