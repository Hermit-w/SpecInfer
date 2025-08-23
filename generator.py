import transformers
import torch
from dataclasses import dataclass
from proposer import ModelProposer, ModelWithCacheProposer
from verifier import Verifier


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
        temperature: float = 0.01,
        attention_mask: torch.Tensor,
    ) -> GeneratorOutput:
        raise NotImplementedError
