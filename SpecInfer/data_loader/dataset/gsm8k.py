import os

from SpecInfer.data_loader.dataset.dataset_wrapper import DatasetWrapper


def transform(i, case):
    _case = {}
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content":  case['question']
        },
        {
            "role": "assistant",
            "content": case['answer']
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = "/export/home/lanliwei.1/datasets"
    dataset_name = "openai/gsm8k"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path, "main")
    dataset.extract_data("train", transform)
