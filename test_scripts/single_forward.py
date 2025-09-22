import os

import torch
import transformers

from SpecInfer import InputForCasualLm, ModelLoader, ModelWithCacheProposer, setup_logger

if __name__ == "__main__":
    # torch.distributed.init_process_group("nccl")
    setup_logger()
    prefix_name = "/export/home/lanliwei.1/models"
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    model_path = os.path.join(prefix_name, model_name)
    
    temperature: float = 1e-5
    rank = 0
    device = torch.device(f"cuda:{rank}")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

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
    # print(inputs.input_ids)
    # print(inputs.attention_mask)
    # import sys
    # sys.exit(1)

    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
 
    # with torch.inference_mode():
    #     generate_config = transformers.GenerationConfig(
    #         num_beams=1,
    #         do_sample=False,
    #         max_new_tokens=200,
    #         temperature=1e-5,
    #         top_k=1,
    #         top_p=0.95,
    #         pad_token_id=151643,
    #         bos_token_id=151643,
    #         eos_token_id=[151645,151643],
    #         )
    #     outputs = model.generate(inputs['input_ids'], attention_mask = inputs['attention_mask'], generation_config=generate_config)
    #     print("".join(tokenizer.batch_decode(outputs)))
    
    # import sys
    # sys.exit(0)

    def sample_method(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor / temperature
        return torch.nn.functional.softmax(tensor, dim=-1)

    with torch.inference_mode():
        outputs = proposer.propose(
            inputs,
            100,
            sample_method,
        )

    if isinstance(outputs, list):
        assert len(outputs) == 1
        outputs = outputs[0]
    
    print(prompt, "".join(tokenizer.batch_decode(outputs.output_ids)))
    
