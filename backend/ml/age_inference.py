# ml/age_inference.py
# -*- coding: utf-8 -*-
"""
Логика вычисления возрастного рейтинга сценариев.

- Берём список сцен (строки текста или dict с полем 'text').
- Для каждой сцены:
    - Лексический анализ (по регуляркам из lexicons.py):
        violence / erotica / profanity / substances / scary
    - Из категорий считаем минимально допустимый возраст для сцены.
    - Если есть обученная нейросетка:
        - Предсказываем возраст для сцены (6 / 12 / 16 / 18)
        - Аккуратно комбинируем с лексическим возрастом (не даём NN
          поднимать чисто детский контент сразу до 16–18).

- На всё произведение:
    - Aggregation по сценам → script_age (лексика), nn_script_age (NN),
      итоговый рейтинг rating_int и rating.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .age_model import AgeClassifier
from .lexicons import (
    VIOLENCE_PATTERNS,
    EROTICA_MILD_PATTERNS,
    EROTICA_HARD_PATTERNS,
    PROFANITY_PATTERNS,
    SUBSTANCES_MILD_PATTERNS,
    SUBSTANCES_HARD_PATTERNS,
    SCARY_PATTERNS,
)

# --------------------------- Константы ---------------------------

# Категории, которыми оперирует сервис и фронтенд
CATEGORIES: List[str] = ["violence", "erotica", "profanity", "substances", "scary"]

# Уровни тяжести
SEVERITY_LABELS = ["none", "mild", "moderate", "severe"]

# Метки возрастных классов нейросети (индекс -> возраст)
# idx: 0 -> 6+, 1 -> 12+, 2 -> 16+, 3 -> 18+
AGE_LABELS: List[int] = [6, 12, 16, 18]


def severity_label(idx: int) -> str:
    if 0 <= idx < len(SEVERITY_LABELS):
        return SEVERITY_LABELS[idx]
    return "none"


# --------------------------- Вспомогательные функции ---------------------------


def _any_match(patterns: List[str], text: str) -> bool:
    """Есть ли совпадение хотя бы по одному паттерну."""
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE | re.MULTILINE):
            return True
    return False


def _scene_to_text(scene: Any) -> str:
    """
    Унификация формата сцены:
    - строка -> как есть
    - dict -> пробуем 'text' / 'scene_text'
    - прочее -> str(scene)
    """
    if isinstance(scene, str):
        return scene
    if isinstance(scene, dict):
        return (
            scene.get("text")
            or scene.get("scene_text")
            or scene.get("content")
            or ""
        )
    return str(scene)


# --------------------------- Лексический анализ ---------------------------


class LexicalAnalyzer:
    """
    Отвечает только за то, чтобы по тексту сцены:
    - определить категории (violence/erotica/… + severity_index)
    - выдать минимальный возраст для сцены по правилам
    """

    def __init__(self) -> None:
        pass

    def detect_categories(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает dict по категориям:

        {
          "violence": {"severity_index": 2, "severity": "moderate", "confidence": 1.0},
          ...
        }
        """
        res: Dict[str, Dict[str, Any]] = {}

        # violence — всё, что в VIOLENCE_PATTERNS считаем хотя бы "moderate"
        vio = 2 if _any_match(VIOLENCE_PATTERNS, text) else 0

        # erotica: mild / hard
        ero_mild = _any_match(EROTICA_MILD_PATTERNS, text)
        ero_hard = _any_match(EROTICA_HARD_PATTERNS, text)
        if ero_hard:
            ero = 3
        elif ero_mild:
            ero = 1
        else:
            ero = 0

        # profanity — все твои матные слова/оскорбления считаем "severe"
        prof = 3 if _any_match(PROFANITY_PATTERNS, text) else 0

        # substances: mild / hard
        sub_mild = _any_match(SUBSTANCES_MILD_PATTERNS, text)
        sub_hard = _any_match(SUBSTANCES_HARD_PATTERNS, text)
        if sub_hard:
            sub = 3
        elif sub_mild:
            sub = 1
        else:
            sub = 0

        # scary — всё, что в SCARY_PATTERNS, считаем "moderate"
        scary = 2 if _any_match(SCARY_PATTERNS, text) else 0

        mapping = {
            "violence": vio,
            "erotica": ero,
            "profanity": prof,
            "substances": sub,
            "scary": scary,
        }

        for cat in CATEGORIES:
            idx = mapping.get(cat, 0)
            res[cat] = {
                "severity_index": idx,
                "severity": severity_label(idx),
                "confidence": 1.0,  # по regex у нас либо 0, либо 1
            }

        return res

    def scene_min_age(self, categories: Dict[str, Dict[str, Any]]) -> int:
        """
        Жёсткие правила конвертации категорий в минимальный возраст сцены.
        Тут как раз можно подкручивать, если слишком завышает.
        """
        age = 6

        vio = categories["violence"]["severity_index"]
        ero = categories["erotica"]["severity_index"]
        prof = categories["profanity"]["severity_index"]
        sub = categories["substances"]["severity_index"]
        scary = categories["scary"]["severity_index"]

        # Насилие
        if vio >= 2:
            age = max(age, 16)

        # Эротика
        if ero == 1:  # мягкая
            age = max(age, 12)
        elif ero >= 3:  # жёсткая
            age = max(age, 18)

        # Мат / оскорбления (у тебя там жёсткие слова)
        if prof >= 1:
            age = max(age, 16)

        # Вещества
        if sub == 1:  # лёгкий алкоголь
            age = max(age, 12)
        elif sub >= 3:  # наркотики / тяжёлые
            age = max(age, 18)

        # Страшилки
        if scary >= 2:
            age = max(age, 12)

        return age


# --------------------------- Сервис с нейросеткой ---------------------------


class AgeRatingService:
    """
    Главный сервис: комбинирует лексику + нейросеть и считает общий рейтинг.
    """

    def __init__(self, model_dir: Optional[str] = None, use_nn: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Использую устройство: {self.device}")

        self.lex = LexicalAnalyzer()

        # флаги и объекты NN
        self.has_age_nn: bool = False
        self.use_nn: bool = use_nn
        self.age_model: Optional[AgeClassifier] = None
        self.age_tokenizer: Optional[AutoTokenizer] = None

        if model_dir is not None and use_nn:
            self._load_age_model(model_dir)
        else:
            print("[INFO] model_dir не передан или use_nn=False, нейросетевая модель возраста отключена.")

    # --------- Загрузка нейросети ---------

    def _load_age_model(self, model_dir: str) -> None:
        """
        Пытается найти чекпоинт в model_dir:
        - сначала: age_checkpoint.pt (наш 4-классовый)
        - далее: best_checkpoint.pt / checkpoint.pt / model.pt
        - если нет — берём первый *.pt
        - если вообще ничего нет — НЕ падаем, просто отключаем NN
        """
        print(f"[AGE_NN] Ищу чекпоинт модели возраста в каталоге: {model_dir}")
        ckpt_candidates = [
            os.path.join(model_dir, "age_checkpoint.pt"),   # наш основной чекпоинт
            os.path.join(model_dir, "best_checkpoint.pt"),
            os.path.join(model_dir, "checkpoint.pt"),
            os.path.join(model_dir, "model.pt"),
        ]

        ckpt_path: Optional[str] = None
        for p in ckpt_candidates:
            if os.path.isfile(p):
                ckpt_path = p
                print(f"[AGE_NN] Найден чекпоинт: {ckpt_path}")
                break

        # если стандартных имён нет — ищем любой *.pt
        if ckpt_path is None and os.path.isdir(model_dir):
            for name in os.listdir(model_dir):
                if name.lower().endswith(".pt"):
                    ckpt_path = os.path.join(model_dir, name)
                    print(f"[AGE_NN] Найден нестандартный чекпоинт: {ckpt_path}")
                    break

        if ckpt_path is None:
            print(
                f"[WARN] В каталоге {model_dir!r} не найдено файлов *.pt. "
                f"Нейросетевая модель возраста будет отключена.\n"
                f"Чтобы включить её снова, запусти обучение, например:\n"
                f"  python -m ml.train_age --csv data/age_scenes.csv "
                f"--epochs 3 --output_dir {model_dir}"
            )
            self.age_model = None
            self.age_tokenizer = None
            self.has_age_nn = False
            return

        print(f"[AGE_NN] Загружаю состояние модели из {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # поддержка разных форматов сохранения
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

        model = AgeClassifier(num_labels=len(AGE_LABELS))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

        self.age_model = model
        self.age_tokenizer = tokenizer
        self.has_age_nn = True

        print("[AGE_NN] Нейросетевая модель возраста успешно загружена.")

    # --------- Батчевое предсказание NN ---------

    def _predict_batch_ages_nn(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Tuple[Optional[int], Optional[float], Optional[int]]]:
        """
        Батчевое предсказание для списка сцен.
        Возвращает список кортежей (age_int, confidence, age_label_idx)
        такой же длины, как texts.
        """
        if not self.use_nn or not self.has_age_nn or self.age_model is None or self.age_tokenizer is None:
            return [(None, None, None)] * len(texts)

        results: List[Tuple[Optional[int], Optional[float], Optional[int]]] = []
        total = len(texts)
        idx = 0

        while idx < total:
            batch_texts = [t for t in texts[idx: idx + batch_size]]

            # если в батче только пустые строки — сразу None
            if all(not t.strip() for t in batch_texts):
                results.extend([(None, None, None)] * len(batch_texts))
                idx += batch_size
                continue

            enc = self.age_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )

            inputs = {
                "input_ids": enc["input_ids"].to(self.device),
                "attention_mask": enc["attention_mask"].to(self.device),
            }

            with torch.no_grad():
                logits = self.age_model(**inputs)  # (batch, num_labels)
                probs = torch.softmax(logits, dim=-1)
                conf_tensors, idx_tensors = torch.max(probs, dim=-1)

            for conf_tensor, idx_tensor, text in zip(conf_tensors, idx_tensors, batch_texts):
                if not text.strip():
                    results.append((None, None, None))
                    continue

                age_label_idx = int(idx_tensor.item())
                confidence = float(conf_tensor.item())
                age_int = AGE_LABELS[age_label_idx]
                results.append((age_int, confidence, age_label_idx))

            print(f"[AGE_NN] Обработано сцен: {min(idx + batch_size, total)} / {total}")
            idx += batch_size

        return results

    # --------- Одиночное предсказание (если вдруг где-то нужно) ---------

    def _predict_scene_age_nn(
        self, text: str
    ) -> Tuple[Optional[int], Optional[float], Optional[int]]:
        """
        Оставил на всякий случай, но в analyze_script мы используем батчевый вариант.
        """
        if not self.use_nn or not self.has_age_nn or self.age_model is None or self.age_tokenizer is None:
            return None, None, None

        if not text.strip():
            return None, None, None

        enc = self.age_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        inputs = {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
        }

        with torch.no_grad():
            logits = self.age_model(**inputs)
            probs = torch.softmax(logits, dim=-1)[0]
            conf_tensor, idx_tensor = torch.max(probs, dim=-1)

        age_label_idx = int(idx_tensor.item())
        confidence = float(conf_tensor.item())
        age_int = AGE_LABELS[age_label_idx]

        return age_int, confidence, age_label_idx

    # --------- Комбинация лексического возраста и NN ---------

    @staticmethod
    def _combine_ages(lex_age: int, nn_age: Optional[int]) -> int:
        """
        Как комбинируем:

        - Если NN нет → возвращаем lex_age.
        - Если NN <= lex_age → доверяем лексике (lex_age).
        - Если NN > lex_age:
            * если lex_age >= 16 → доверяем NN полностью;
            * если lex_age == 12 → позволяем поднять максимум до 16;
            * если lex_age == 6 → позволяем поднять максимум до 12
              (никогда не делаем 16/18 только из-за NN на детском тексте).
        """
        if nn_age is None:
            return lex_age

        if nn_age <= lex_age:
            return lex_age

        if lex_age >= 16:
            return nn_age

        if lex_age == 12:
            return min(nn_age, 16)

        if lex_age == 6:
            return min(nn_age, 12)

        return max(lex_age, nn_age)

    # --------- Публичный метод анализа ---------

        # --------- Публичный метод анализа ---------

    def analyze_script(
        self, scenes: List[Any], filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Главный метод, который зовёт сервис:

        scenes — список сцен (строка или dict с 'text').

        Делает:
        - нормализацию списка сцен (обрезаем мусор, слишком короткие куски)
        - при необходимости сжимаем слишком большое количество сцен в более крупные блоки

        Возвращает dict:
        {
          "rating": "12+",
          "rating_int": 12,
          "scenes_total": ...,
          "scenes_with_violations": ...,
          "per_category": {...},
          "problem_scenes": [...],
          "script_age": ...,
          "lex_age": ...,
          "nn_script_age": ...,
          "scene_results": [...]
        }
        """
        # ---------- НОРМАЛИЗАЦИЯ СЦЕН ----------

        MIN_SCENE_LEN = 30       # как в build_age_dataset
        MAX_SCENES = 2000        # жёстный верхний предел, чтобы не убить моделью машину

        # 1) приводим всё к тексту + выкидываем пустое/очень короткое
        normalized_scenes: List[str] = []
        for scene in scenes:
            text = _scene_to_text(scene).strip()
            if len(text) < MIN_SCENE_LEN:
                continue
            normalized_scenes.append(text)

        if not normalized_scenes:
            # вообще ничего адекватного не нашли
            return {
                "rating": "6+",
                "rating_int": 6,
                "scenes_total": 0,
                "scenes_with_violations": 0,
                "per_category": {cat: {
                    "max_severity_index": 0,
                    "max_severity": severity_label(0),
                    "episodes": 0,
                    "scene_percent": 0.0,
                } for cat in CATEGORIES},
                "problem_scenes": [],
                "script_age": 6,
                "lex_age": 6,
                "nn_script_age": None,
                "scene_results": [],
                **({"filename": filename} if filename is not None else {}),
            }

        # 2) если сцен ОЧЕНЬ много — сжимаем в более крупные блоки
        if len(normalized_scenes) > MAX_SCENES:
            print(f"[AGE] Слишком много сцен ({len(normalized_scenes)}). "
                  f"Объединяю их примерно в {MAX_SCENES} блоков.")
            chunk_size = (len(normalized_scenes) + MAX_SCENES - 1) // MAX_SCENES
            merged: List[str] = []
            buf: List[str] = []
            for i, t in enumerate(normalized_scenes, 1):
                buf.append(t)
                if i % chunk_size == 0:
                    merged.append("\n\n".join(buf))
                    buf = []
            if buf:
                merged.append("\n\n".join(buf))
            normalized_scenes = merged

        # Теперь scenes — уже нормализованный список
        scenes = normalized_scenes

        # ---------- ДАЛЬШЕ СТАРАЯ ЛОГИКА, ТОЛЬКО ЧУТЬ ПОДРЕЗАННАЯ ПОД НОВЫЕ scenes ----------

        scene_results: List[Dict[str, Any]] = []

        lex_scene_ages: List[int] = []
        nn_scene_ages: List[int] = []

        for i, text in enumerate(scenes):
            # 1) Лексика
            cats = self.lex.detect_categories(text)
            lex_age = self.lex.scene_min_age(cats)
            lex_scene_ages.append(lex_age)

            # 2) Нейросеть
            nn_age_int, nn_conf, nn_idx = self._predict_scene_age_nn(text)
            if nn_age_int is not None:
                nn_scene_ages.append(nn_age_int)

            # 3) Комбинированный возраст для сцены
            combined_age = self._combine_ages(lex_age, nn_age_int)

            # confidence у возраста сцены: либо NN, либо 1.0 (если NN нет)
            age_conf = float(nn_conf) if nn_conf is not None else 1.0
            age_idx_for_scene = (
                nn_idx if nn_idx is not None else AGE_LABELS.index(combined_age)
            )

            scene_results.append(
                {
                    "categories": cats,
                    "scene_age": combined_age,
                    "age_confidence": age_conf,
                    "age_label_idx": int(age_idx_for_scene),
                    "scene_id": i,
                    "text_snippet": text[:1000],
                }
            )

        # --------- Агрегация по всему сценарию ---------

        scenes_total = len(scene_results)
        scenes_with_violations = sum(
            1
            for s in scene_results
            if any(
                s["categories"][cat]["severity_index"] > 0 for cat in CATEGORIES
            )
        )

        # per_category
        per_category: Dict[str, Dict[str, Any]] = {}
        for cat in CATEGORIES:
            severities = [s["categories"][cat]["severity_index"] for s in scene_results]
            max_sev = max(severities) if severities else 0
            episodes = sum(1 for v in severities if v > 0)
            scene_percent = float(episodes / scenes_total) if scenes_total > 0 else 0.0

            per_category[cat] = {
                "max_severity_index": max_sev,
                "max_severity": severity_label(max_sev),
                "episodes": episodes,
                "scene_percent": scene_percent,
            }

                # script_age и lex_age — чисто по лексике (максимум по сценам)
        lex_age_script = max(lex_scene_ages) if lex_scene_ages else 6
        script_age = lex_age_script

        # nn_script_age — максимум предсказаний NN по сценам (если есть)
        nn_script_age: Optional[int]
        if nn_scene_ages:
            nn_script_age = max(nn_scene_ages)
        else:
            nn_script_age = None

        # итоговый возраст сценария (combination)
        final_age_int = self._combine_ages(script_age, nn_script_age)

        # 🔥 ВАЖНО: если лексика говорит "чистый 6+" и НЕТ нарушений,
        # не даём нейросети задирать рейтинг
        if script_age == 6 and scenes_with_violations == 0:
            final_age_int = 6

        rating_str = f"{final_age_int}+"

        # problem_scenes — все сцены, где есть нарушения (severity > 0),
        # плюс сцены с возрастом > 6
        problem_scenes: List[Dict[str, Any]] = []
        for s in scene_results:
            has_violation = any(
                s["categories"][cat]["severity_index"] > 0 for cat in CATEGORIES
            )
            if has_violation or s["scene_age"] > 6:
                issues = []
                for cat in CATEGORIES:
                    sev_idx = s["categories"][cat]["severity_index"]
                    if sev_idx > 0:
                        issues.append(
                            {
                                "category": cat,
                                "severity": s["categories"][cat]["severity"],
                                "severity_index": sev_idx,
                                "confidence": s["categories"][cat]["confidence"],
                            }
                        )
                problem_scenes.append(
                    {
                        "scene_id": s["scene_id"],
                        "scene_age": s["scene_age"],
                        "issues": issues,
                        "text_snippet": s["text_snippet"],
                    }
                )

        result: Dict[str, Any] = {
            "rating": rating_str,
            "rating_int": final_age_int,
            "scenes_total": scenes_total,
            "scenes_with_violations": scenes_with_violations,
            "per_category": per_category,
            "problem_scenes": problem_scenes,
            "script_age": script_age,
            "lex_age": lex_age_script,
            "nn_script_age": nn_script_age,
            "scene_results": scene_results,
        }

        if filename is not None:
            result["filename"] = filename

        return result

