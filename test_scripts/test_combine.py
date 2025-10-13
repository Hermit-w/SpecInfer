import os

import torch

from SpecInfer import (InputForCasualLm, ModelLoader, ModelWithCacheProposer,
                       Verifier, get_softmax_func, setup_logger, speculative_sample)

if __name__ == "__main__":
    setup_logger(level="debug")
    torch.distributed.init_process_group("nccl")
    prefix_name = "/export/home/lanliwei.1/models"
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    draft_model_name = "Qwen/Qwen3-0.6B"
    model_path = os.path.join(prefix_name, model_name)
    draft_model_path = os.path.join(prefix_name, draft_model_name)
    tp_size = 2
    model, tokenizer = ModelLoader.load_model_and_tokenizer(model_path, tp_size)
    draft_model = ModelLoader.load_model(draft_model_path, tp_size)
    temperature = 1e-5
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{rank}")
    proposer = ModelWithCacheProposer(draft_model, tokenizer)
    verifier = Verifier(model, tokenizer)
    prompt = "Hello, please introduce yourself.\n"

    draft_inputs = InputForCasualLm.from_prompt(prompt, tokenizer).to(device)
    target_inputs = InputForCasualLm.from_prompt(prompt, tokenizer).to(device)

    sample_method = get_softmax_func(temperature)

    for _ in range(2):
        draft_outputs = proposer.propose(
            draft_inputs,
            6,
            sample_method
        )

        target_outputs = verifier.verify(target_inputs, draft_outputs, sample_method)

        print(target_outputs)

        accept_tokens = speculative_sample(draft_outputs, target_outputs)

        draft_inputs = proposer.adjust_input(accept_tokens, draft_inputs, draft_outputs)
        print(draft_inputs)
        target_inputs = verifier.adjust_input(accept_tokens, target_outputs)
        print(target_inputs)

    torch.distributed.destroy_process_group()
