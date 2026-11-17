from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import KucTokenizer


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


class SafetyDataset(Dataset):
    def __init__(self, path: str | Path, tok: KucTokenizer, max_len: int) -> None:
        self.rows = read_jsonl(path)
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        ids = self.tok.encode_prompt(row["text"], self.max_len)
        return {
            "input_ids": ids,
            "answer_id": self.tok.answer_ids[int(row["label"])],
            "label": int(row["label"]),
        }


def collate(batch: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    answer_ids = torch.tensor([item["answer_id"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return input_ids, answer_ids, labels


class Batcher:
    """Token-budget aware batches so long inputs do not blow up VRAM."""

    def __init__(self, dataset: SafetyDataset, batch_size: int, max_tokens: int, shuffle: bool, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            self.rng.shuffle(order)
        batch: list[dict] = []
        tokens = 0
        longest = 0
        for index in order:
            item = self.dataset[index]
            length = len(item["input_ids"])
            new_longest = max(longest, length)
            if batch and ((len(batch) + 1) * new_longest > self.max_tokens or len(batch) >= self.batch_size):
                yield batch
                batch, tokens, longest = [], 0, 0
                new_longest = length
            batch.append(item)
            longest = new_longest
            tokens = len(batch) * longest
        if batch:
            yield batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
