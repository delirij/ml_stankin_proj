"""
doc_to_scenes.py

Извлечение текста из DOCX (и частично DOC) и разбиение на сцены.
Без textract, только python-docx.
"""

from __future__ import annotations

from io import BytesIO
from typing import List

from docx import Document  # pip install python-docx

# переиспользуем ту же логику разбиения, что и для PDF
from .pdf_to_scenes import split_into_scenes


def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """
    Извлекает чистый текст из DOCX-файла (байты).
    Возвращает один большой строковый текст с переносами строк.
    """
    if not file_bytes:
        return ""

    try:
        doc = Document(BytesIO(file_bytes))
    except Exception:
        # если что-то пошло не так, просто вернём пустую строку,
        # чтобы сервис не падал
        return ""

    paragraphs: List[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            paragraphs.append(text)

    # иногда полезно добавлять пустую строку между абзацами,
    # но для сценариев чаще достаточно обычных переносов
    return "\n".join(paragraphs)


__all__ = [
    "extract_text_from_docx_bytes",
    "split_into_scenes",
]
