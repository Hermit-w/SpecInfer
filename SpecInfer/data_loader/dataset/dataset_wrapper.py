import json
import random
from typing import Callable, Optional, Union

from datasets import load_dataset


class DatasetWrapper:
    def __init__(
        self,
        dataset_name_or_path: str,
        *args,
        **kwargs
    ):
        self.dataset = load_dataset(
            dataset_name_or_path,
            *args,
            **kwargs
        )
        self.dataset_name = "_".join(dataset_name_or_path.split("/")[-2:])

    def get_raw_data(
        self,
    ):
        return self.dataset

    def get_raw_filename(self, split: str, prefix: Optional[str]):
        if prefix is not None:
            return f"data/{prefix}_{self.dataset_name}_{split}_raw.json"
        else:
            return f"data/{self.dataset_name}_{split}_raw.json"

    def get_output_filename(self, split: str, prefix: Optional[str]):
        if prefix is not None:
            return f"data/{prefix}_{self.dataset_name}_{split}.json"
        else:
            return f"data/{self.dataset_name}_{split}.json"

    def extract_data(
        self,
        splits: Union[str, list[str]],
        transform: Callable[[int, dict[str, str]], dict[str, list[str]]],
        size: int = -1,
        prefix: Optional[str] = None,
    ):
        splits = [splits] if isinstance(splits, str) else splits
        cases = []
        for split in splits:
            rawdata = self.dataset[split]
            for idx, case in enumerate(rawdata):
                cases.append(transform(idx, case))

        if size > 0:
            random.shuffle(cases)
            cases = cases[:size]
        output_file = self.get_output_filename(split, prefix)

        with open(output_file, 'w') as f:
            json.dump(cases, f, indent=4)

        print(f"Dataset: {self.dataset_name}, Split: {split}, Length: {len(cases)}")