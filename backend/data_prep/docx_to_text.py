# data_prep/docx_to_text.py
# -*- coding: utf-8 -*-

from typing import List
import io

from docx import Document  # нужно: pip install python-docx


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    """
    Извлекает текст из DOCX-байтов с помощью python-docx.
    """
    bio = io.BytesIO(docx_bytes)
    doc = Document(bio)
    paragraphs: List[str] = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)
