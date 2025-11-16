# ml/train_age.py
# -*- coding: utf-8 -*-

import os
import argparse

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW

from .age_model import AgeClassifier
from data_prep.build_age_dataset import AGE2IDX, IDX2AGE


class AgeScenesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Использую устройство: {device}")

    df = pd.read_csv(args.csv)
    print(f"Загружено строк: {len(df)} из {args.csv}")

    base_model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    dataset = AgeScenesDataset(df, tokenizer, max_length=args.max_length)

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = AgeClassifier(base_model_name=base_model_name, num_labels=len(AGE2IDX)).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            logits = model(**batch)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")

                logits = model(**batch)
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item() * labels.size(0)

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_ds)
        val_acc = correct / max(1, total)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        # сохраняем лучший чекпоинт
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "base_model_name": base_model_name,
                "num_labels": len(AGE2IDX),
                "age2idx": AGE2IDX,
                "idx2age": IDX2AGE,
            }
            torch.save(ckpt, os.path.join(args.output_dir, "age_checkpoint.pt"))
            tokenizer.save_pretrained(args.output_dir)
            print(f"  >> Сохранён лучший чекпоинт в {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/age_scenes.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="models/age_model")
    args = parser.parse_args()
    train(args)
