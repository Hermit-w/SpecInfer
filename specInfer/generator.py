import copy
import logging
from dataclasses import dataclass

import torch
import transformers
from common import (InputForCasualLm, OutputForCasualLm, synchronize_time,
                    target_sample_from_distribution)
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
    
    def speculative_sample(
        self,
        proposed_output: OutputForCasualLm,
        verified_output: OutputForCasualLm
    ) -> tuple[list[torch.Tensor], list[float], list[int]]:
        # Accept-reject token loop
        accept_ids: list[torch.Tensor] = []
        if proposed_output.output_ids is None:
            logger.error("Get none output_ids!")
            raise ValueError
        bs = proposed_output.output_ids.shape[0]
        sample_steps: list[int] = [0] * bs
        alpha: list[float] = [0.0] * bs
        for i in range(bs):
            all_accepted: bool = True
            accept_ids_i: list[torch.Tensor] = []
            for j in range(proposed_output.generated_length):
                sampled_ratios = (
                    verified_output.output_distribution[i, j,proposed_output.output_ids[i, j]]
                    / proposed_output.output_distribution[i, j, proposed_output.output_ids[i, j]]
                )

                sampled_ratios = torch.min(sampled_ratios,
                                           torch.ones_like(sampled_ratios))
                rs = torch.rand_like(sampled_ratios)
                # logger.log("sample ratio", (rs, sampled_ratios))
                cur_alpha_tensor = torch.min(verified_output.output_distribution[i, j, proposed_output.output_ids[i, j]],
                                             proposed_output.output_distribution[i, j, proposed_output.output_ids[i, j]])
                assert cur_alpha_tensor.numel() == 1
                cur_alpha = float(cur_alpha_tensor)


                assert cur_alpha >= 0 and cur_alpha <= 1
                alpha[i] += cur_alpha
                sample_steps[i] += 1
                if rs < sampled_ratios:
                    accept_ids_i.append(proposed_output.output_ids[i, j].unsqueeze(0))
                else:
                    all_accepted = False
                    next_token_id = target_sample_from_distribution(
                        verified_output.output_distribution[i, j, :],
                        proposed_output.output_distribution[i, j, :]
                    )
                    accept_ids_i.append(next_token_id.unsqueeze(0))
                    break

            # if all tokens were accepted, sample a last one
            if all_accepted:
                next_token_id = torch.multinomial(
                    verified_output.output_distribution[i, -1, :],
                    num_samples=1,
                )

                assert next_token_id.dim() == 1
                accept_ids_i.append(next_token_id)
            accept_ids.append(torch.cat(accept_ids_i, dim=0))

        return accept_ids, alpha, sample_steps

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
        batch_size = input_ids.shape[0]

        def logits_to_scores(logits: torch.Tensor):
            return torch.softmax(logits / temperature, dim=-1)
        generated_token_cnt: int = 0
        generated_tokens = None
        wrong_token_ids: list = []

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
                logits_to_scores
            )
            propose_steps += 1

            # prepare verifier input
            verifier_input = self.verifier.prepare_input(
                proposer_output,
                verifier_input
            )
            logger.debug(verifier_input.input_ids)

            # forward n tokens on the model in the a single run
            verifier_output = self.verifier.verify(
                verifier_input,
                proposer_output.generated_length,
                logits_to_scores
            )

            # compare selected tokens
            # accept_token_ids, cur_alpha, cur_sample_steps = self.compare_tokens(proposer_output, verifier_output)
            accept_token_ids, cur_alpha, cur_sample_steps = self.speculative_sample(
                proposer_output,
                verifier_output
            )
            logger.debug(accept_token_ids)

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

            if (max_tokens > 0 and generated_token_cnt >= max_tokens) \
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
