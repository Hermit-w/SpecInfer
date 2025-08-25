import transformers
from datasets import Dataset, DatasetDict, load_dataset
import torch
from specInfer.generator import SpecGenerator
from specInfer import logger

def get_inputs(
        dataset_path: str,
        name: str,
        tokenizer: transformers.PreTrainedTokenizerBase
    ):
    data: DatasetDict = load_dataset(dataset_path, name)
    data_train: Dataset = data["train"]
    data_test: Dataset = data["test"]
    
    for i in data_train:
        origin_input = tokenizer(i["question"])
        yield {'input_ids': torch.tensor(origin_input["input_ids"], device=torch.device("cuda:0")).unsqueeze_(0),
               'attention_mask': torch.tensor(origin_input["attention_mask"], device=torch.device("cuda:0")).unsqueeze_(0)}


if __name__ == '__main__':
    logger.setup_logger(
        level='debug'
    )
    
    torch.manual_seed(12)
    draft_model_path: str = "/export/home/lanliwei.1/models/Qwen/Qwen3-0.6B"
    target_model_path: str = "/export/home/lanliwei.1/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
    dataset_path: str = "/export/home/lanliwei.1/datasets/openai/gsm8k"

    draft_model = transformers.AutoModelForCausalLM.from_pretrained(draft_model_path)
    draft_model.cuda()
    target_model = transformers.AutoModelForCausalLM.from_pretrained(draft_model_path)
    target_model.cuda()

    draft_tokenizer = transformers.AutoTokenizer.from_pretrained(draft_model_path)
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(target_model_path)

    generateor = SpecGenerator(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=target_tokenizer,
        max_propose_num=6,
        use_cache=True,
        benchmark_time=False,
    )

    for idx, inputs in enumerate(get_inputs(dataset_path, "main", target_tokenizer)):
        # print(input)
        # break
        output = generateor.generate(
            **inputs,
            max_tokens=-1,
            temperature=0.01,
        )
        print("".join(output.output))
        print(output)
        break