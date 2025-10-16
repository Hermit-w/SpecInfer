import json
import os

MAPPING = {
    "gsm8k": "openai_gsm8k_train.json",
}


class DatasetLoader:
    @classmethod
    def load_dataset(
        cls,
        name: str,
        path: str,
    ) -> list[dict]:
        file = MAPPING[name]
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
