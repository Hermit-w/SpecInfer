import os

from SpecInfer.data_loader.dataset.dataset_wrapper import DatasetWrapper


def transform(i, case):
    _case = {}
    SQL_prompt = "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. "
    _case["id"] = i
    _case["conversation"] = [
        {
            "role": "user",
            "content": SQL_prompt + case['question']
        },
        {
            "role": "assistant",
            "content": " ".join(case['query_toks_no_value'])
        }
    ]
    return _case


if __name__ == '__main__':
    prefix_name = "/export/home/lanliwei.1/datasets"
    dataset_name = "xlangai/spider"
    dataset_path = os.path.join(prefix_name, dataset_name)
    dataset = DatasetWrapper(dataset_path)
    dataset.extract_data("train", transform)
