import json
import os

import torch
from datasets import Dataset


class AlpacaDatasetProcessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_instruction(
        self, instruction: str, input_text: str = "", output: str = ""
    ) -> str:
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"  # noqa: E501
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt

    def process_function(self, examples):
        prompts = [
            self.format_instruction(
                instruction=instr, input_text=inp if inp else "", output=out
            )
            for instr, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]

        encodings = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        encodings["labels"] = encodings["input_ids"].clone()
        return encodings


class AlpacaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        labels = self.encodings["labels"][idx]

        model_inputs = torch.stack([input_ids, attention_mask, labels])
        target = torch.tensor([0.0])
        return model_inputs, target

    def __len__(self):
        return len(self.encodings["input_ids"])


def load_alpaca_dataset():
    local_path = "./artifacts/alpaca_data.json"

    if not os.path.exists(local_path):
        print("Downloading Alpaca dataset...")
        import requests

        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        response = requests.get(url)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(response.text)

    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_dict = {"instruction": [], "input": [], "output": []}

    for item in data:
        dataset_dict["instruction"].append(item["instruction"])
        dataset_dict["input"].append(item["input"])
        dataset_dict["output"].append(item["output"])

    return Dataset.from_dict(dataset_dict)


def prepare_datasets(tokenizer, max_length: int = 512):
    dataset = load_alpaca_dataset()
    dataset = dataset.train_test_split(test_size=0.1)

    processor = AlpacaDatasetProcessor(tokenizer, max_length)

    train_encodings = processor.process_function(dataset["train"])

    train_dataset = AlpacaDataset(train_encodings)

    return train_dataset
