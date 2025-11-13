"""
Offline training routine for ADRIAN's intent classifier.

This script demonstrates how to fine-tune a lightweight transformer
(`distilbert-base-uncased`) on a custom dataset of labeled intents.

Usage (example):
    python services/processing_service/train_intent_classifier.py \\
        --data-file data/intent_examples.jsonl \\
        --output-dir models/intent_classifier

This file is intentionally lightweight and serves as documentation plus a
starting point. It is not invoked by ADRIAN services at runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

INTENT_LABELS = [
    "system_control",
    "search",
    "task_management",
    "conversation",
]


@dataclass
class TrainingConfig:
    data_file: Path
    output_dir: Path
    model_name: str = "distilbert-base-uncased"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_length: int = 256
    seed: int = 42


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train ADRIAN intent classifier.")
    parser.add_argument("--data-file", type=Path, required=True, help="Path to JSONL dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    return TrainingConfig(
        data_file=args.data_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


def prepare_dataset(config: TrainingConfig):
    """
    Expect data-file as JSONL with fields: {"text": "...", "intent": "..."}.
    """
    dataset = load_dataset("json", data_files=str(config.data_file))

    label2id = {label: idx for idx, label in enumerate(INTENT_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )
        tokens["labels"] = [label2id.get(intent, label2id["conversation"]) for intent in batch["intent"]]
        return tokens

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text", "intent"])
    tokenized.set_format(type="torch")

    return tokenized, tokenizer, label2id, id2label


def train(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)

    dataset, tokenizer, label2id, id2label = prepare_dataset(config)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    train_loader = DataLoader(dataset["train"], shuffle=True, batch_size=config.batch_size)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = config.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(config.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(config.num_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1} loss {loss.item():.4f}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Model saved to {config.output_dir}")


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()

