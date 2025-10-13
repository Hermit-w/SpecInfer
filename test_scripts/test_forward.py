import os
import logging
import torch

from SpecInfer import InputForCasualLm, ModelLoader, ModelWithCacheProposer, setup_logger

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")
    setup_logger()
    prefix_name = "/export/home/lanliwei.1/models"
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    model_path = os.path.join(prefix_name, model_name)
    tp_size = 2
    model, tokenizer = ModelLoader.load_model_and_tokenizer(model_path, tp_size=tp_size)
    temperature: float = 1e-5
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{rank}")
    logger.info(f"At rank: {rank}")

    proposer = ModelWithCacheProposer(
        model,
        tokenizer,
        False,
    )

    prompt = "Hello, please introduce yourself.\n"

    inputs = InputForCasualLm.from_prompt(
        prompt,
        tokenizer
    )

    def sample_method(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor / temperature
        return torch.nn.functional.softmax(tensor, dim=-1)

    outputs = proposer.propose(
        inputs,
        200,
        sample_method,
    )

    if isinstance(outputs, list):
        assert len(outputs) == 1
        outputs = outputs[0]

    if rank == 0:
        logger.info(prompt + "".join(tokenizer.batch_decode(outputs.output_ids)))
    torch.distributed.destroy_process_group()
