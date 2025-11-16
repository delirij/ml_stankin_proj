# data_prep/build_age_dataset.py
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import pandas as pd

from .pdf_to_scenes import extract_text_from_pdf_bytes, split_into_scenes
from .doc_to_scenes import extract_text_from_docx_bytes

# Мы будем учить 4 класса: 6, 12, 16, 18
AGE_CLASSES = [6, 12, 16, 18]

AGE2IDX = {
    6: 0,
    12: 1,
    16: 2,
    18: 3,
}
IDX2AGE = {v: k for k, v in AGE2IDX.items()}


def age_from_dirname(dirname: str) -> int:
    """
    Извлекает возраст из названия папки, например:
    'Доктор Вера 18+' -> 18
    'Любопытная Варвара 6+' -> 6
    """
    m = re.search(r"(\d+)\+", dirname)
    if not m:
        raise ValueError(f"Не удалось найти возраст в названии папки: {dirname}")
    age = int(m.group(1))
    if age not in AGE2IDX:
        raise ValueError(f"Возраст {age}+ пока не поддерживается (ожидаем один из {AGE_CLASSES})")
    return age


def build_age_dataset(root_dir: str = "data/labeled_scripts", out_csv: str = "data/age_scenes.csv"):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Папка с размеченными сценариями не найдена: {root_dir}")

    rows = []

    # обходим папки вида 'Доктор Вера 18+', 'Любопытная Варвара 6+' и т.п.
    for show_dir in root.iterdir():
        if not show_dir.is_dir():
            continue

        try:
            age = age_from_dirname(show_dir.name)
        except ValueError as e:
            print(f"[WARN] {e}, пропускаю папку {show_dir}")
            continue

        print(f"Обрабатываю шоу: {show_dir.name} (возраст {age}+)")

        # рекурсивно обходим все файлы внутри
        for f in show_dir.rglob("*"):
            if not f.is_file():
                continue

            suffix = f.suffix.lower()
            if suffix not in [".pdf", ".docx"]:
                # .doc, .txt и прочее игнорируем
                print(f"  [SKIP] {f.name} (расширение {suffix} не поддерживается)")
                continue

            print(f"  Читаю файл: {f.relative_to(root)}")

            raw_bytes = f.read_bytes()
            if suffix == ".pdf":
                text = extract_text_from_pdf_bytes(raw_bytes)
            else:  # .docx
                text = extract_text_from_docx_bytes(raw_bytes)

            scenes = split_into_scenes(text)

            for scene_id, scene_text in enumerate(scenes):
                scene_text = scene_text.strip()
                if len(scene_text) < 30:
                    # слишком короткие сцены выкидываем
                    continue

                rows.append(
                    {
                        "script": str(f.relative_to(root)),
                        "scene_id": scene_id,
                        "text": scene_text,
                        "age": age,
                        "label": AGE2IDX[age],
                    }
                )

    df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Сохранён датасет сцен {out_path} (строк: {len(df)})")


if __name__ == "__main__":
    build_age_dataset()
