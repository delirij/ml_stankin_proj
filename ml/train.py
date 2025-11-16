# ml/train_age.py
# -*- coding: utf-8 -*-

"""
Тренировка модели возрастного рейтинга на 4 класса: 6 / 12 / 16 / 18.

Ожидается CSV с колонками:
    - text : текст сцены/фрагмента
    - age  : одно из значений {6, 12, 16, 18}

Пример запуска (как у тебя в age_inference.py в подсказке):

    python -m ml.train_age ^
        --csv data/age_scenes.csv ^
        --output_dir models/age_model ^
        --epochs 3 ^
        --batch_size 8

В результате в output_dir будет:
    checkpoint.pt

AgeRatingService из age_inference.py автоматически его подхватит.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .age_model import AgeClassifier  # твоя NN-модель


# ======================== Константы ========================

# Те же метки, что и в age_inference.py
AGE_LABELS: List[int] = [6, 12, 16, 18]
AGE_TO_IDX: Dict[int, int] = {age: i for i, age in enumerate(AGE_LABELS)}


# ======================== Датасет ========================


class AgeScenesDataset(Dataset):
    """
    Датасет для обучения модели возраста.

    CSV формат:
        text,age
        "Какой-то текст сцены",6
        "Другой текст",12
        ...
    """

    def __init__(self, csv_path: str, tokenizer_name: str, max_len: int = 256):
        self.df = pd.read_csv(csv_path)

        if "text" not in self.df.columns or "age" not in self.df.columns:
            raise ValueError("CSV должен содержать колонки 'text' и 'age'")

        # фильтруем только те строки, где возраст входит в допустимый список
        self.df = self.df[self.df["age"].isin(AGE_TO_IDX.keys())].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                "В CSV нет строк с age в {6, 12, 16, 18}. "
                "Проверь разметку данных."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row["text"])
        age = int(row["age"])

        label_idx = AGE_TO_IDX[age]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),  # [seq_len]
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }
        return item


# ======================== Train / Eval ========================


def train_one_epoch(
    model: AgeClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


@torch.no_grad()
def eval_one_epoch(
    model: AgeClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> (float, float):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


# ======================== main() ========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение модели возрастного рейтинга на 4 класса (6/12/16/18)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Путь к CSV с колонками 'text' и 'age'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Куда сохранить checkpoint.pt (например, models/age_model).",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="DeepPavlov/rubert-base-cased",
        help="Имя базовой BERT-модели для токенизатора и AgeClassifier (если он её использует).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Доля шагов под warmup в линейном scheduler’е.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Использую устройство: {device}")

    # --- датасет ---
    dataset = AgeScenesDataset(
        csv_path=args.csv,
        tokenizer_name=args.pretrained_model,
        max_len=args.max_len,
    )

    # делим на train/val (90/10)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # на Windows лучше 0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --- модель ---
    model = AgeClassifier(num_labels=len(AGE_LABELS))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio) if total_steps > 0 else 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = None
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n======== Эпоха {epoch}/{args.epochs} ========")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"[Train] loss = {train_loss:.4f}")

        val_loss, val_acc = eval_one_epoch(model, val_loader, device)
        print(f"[Val]   loss = {val_loss:.4f}, acc = {val_acc:.4f}")

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

    if best_state_dict is None:
        print("[WARN] best_state_dict пустой, сохраняю текущие веса модели.")
        best_state_dict = model.state_dict()

    ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")
    torch.save({"model_state_dict": best_state_dict}, ckpt_path)
    print(f"[OK] Чекпоинт сохранён в {ckpt_path}")


if __name__ == "__main__":
    main()
