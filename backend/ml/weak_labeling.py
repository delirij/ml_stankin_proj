from pathlib import Path
from typing import Dict, List
import re

import pandas as pd

from data_prep.pdf_to_scenes import extract_text_from_pdf_bytes, split_into_scenes
from data_prep.doc_to_scenes import extract_text_from_docx_bytes
from ml.config import CATEGORIES
from ml.lexicons import (
    VIOLENCE_PATTERNS,
    EROTICA_MILD_PATTERNS,
    EROTICA_HARD_PATTERNS,
    PROFANITY_PATTERNS,
    SUBSTANCES_MILD_PATTERNS,
    SUBSTANCES_HARD_PATTERNS,
    SCARY_PATTERNS,
)


def count_matches(patterns: List[str], text: str) -> int:
    cnt = 0
    for p in patterns:
        cnt += len(re.findall(p, text, flags=re.IGNORECASE | re.UNICODE))
    return cnt


def severity_from_count(cnt: int) -> int:
    """
    Базовая шкала:
    0: нет совпадений
    1: 1–2
    2: 3–5
    3: >5
    """
    if cnt == 0:
        return 0
    elif cnt <= 2:
        return 1
    elif cnt <= 5:
        return 2
    else:
        return 3


def label_scene(text: str) -> Dict[str, int]:
    """
    Возвращает severity 0..3 по каждой категории.
    Жёсткая эротика/наркотики -> сразу 3 (18+ по этой шкале).
    """

    # Насилие, мат, страшное — обычная шкала.
    violence_cnt = count_matches(VIOLENCE_PATTERNS, text)
    profanity_cnt = count_matches(PROFANITY_PATTERNS, text)
    scary_cnt = count_matches(SCARY_PATTERNS, text)

    violence_sev = severity_from_count(violence_cnt)
    profanity_sev = severity_from_count(profanity_cnt)
    scary_sev = severity_from_count(scary_cnt)

    # Эротика: мягкая + жёсткая
    erotica_mild_cnt = count_matches(EROTICA_MILD_PATTERNS, text)
    erotica_hard_cnt = count_matches(EROTICA_HARD_PATTERNS, text)

    if erotica_hard_cnt > 0:
        erotica_sev = 3
    else:
        erotica_sev = severity_from_count(erotica_mild_cnt)
        erotica_sev = min(erotica_sev, 2)  # мягкая эротика максимум до 2 (12/16+)

    # Вещества: мягкий алкоголь + жёсткие наркотики
    substances_mild_cnt = count_matches(SUBSTANCES_MILD_PATTERNS, text)
    substances_hard_cnt = count_matches(SUBSTANCES_HARD_PATTERNS, text)

    if substances_hard_cnt > 0:
        substances_sev = 3
    else:
        substances_sev = severity_from_count(substances_mild_cnt)
        substances_sev = min(substances_sev, 2)  # алкоголь максимум 2

    return {
        "violence": violence_sev,
        "erotica": erotica_sev,
        "profanity": profanity_sev,
        "substances": substances_sev,
        "scary": scary_sev,
    }


def build_pseudo_labeled_csv(scripts_dir: str, out_csv: str):
    """
    Берём все PDF и DOCX из папки scripts_dir, режем на сцены, размечаем словарями
    и сохраняем в CSV, который потом идёт в обучение нейросети.
    """
    rows = []
    scripts_path = Path(scripts_dir)

    pdf_paths = list(scripts_path.glob("*.pdf"))
    docx_paths = list(scripts_path.glob("*.docx"))

    all_paths = pdf_paths + docx_paths

    if not all_paths:
        print(f"В папке {scripts_dir} нет PDF/DOCX-файлов")
        return

    for script_path in all_paths:
        print(f"Обрабатываю {script_path.name}")
        raw_bytes = script_path.read_bytes()

        # PDF или DOCX
        if script_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf_bytes(raw_bytes)
        elif script_path.suffix.lower() == ".docx":
            text = extract_text_from_docx_bytes(raw_bytes)
        else:
            # На случай, если вдруг затесалось что-то ещё.
            print(f"Пропускаю {script_path.name} (не .pdf и не .docx)")
            continue

        scenes = split_into_scenes(text)

        for scene_id, scene_text in enumerate(scenes):
            labels = label_scene(scene_text)
            row = {
                "script": script_path.name,
                "scene_id": scene_id,
                "text": scene_text,
            }
            for cat in CATEGORIES:
                row[cat] = labels[cat]
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Сохранён псевдодатасет {out_path} (строк: {len(df)})")


if __name__ == "__main__":
    # ВАЖНО: сюда клади и .pdf, и .docx сценарии
    build_pseudo_labeled_csv(scripts_dir="scripts_pdf", out_csv="data/train_pseudo.csv")
