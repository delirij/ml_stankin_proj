# data_prep/pdf_to_scenes.py
# -*- coding: utf-8 -*-

from typing import List
import io
import re

from pdfminer.high_level import extract_text


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Извлекает текст из PDF-байтов с помощью pdfminer.six.
    """
    bio = io.BytesIO(pdf_bytes)
    text = extract_text(bio)
    return text or ""


def split_into_scenes(
    text: str,
    min_scene_chars: int = 300,
    max_scene_chars: int = 3000,
) -> List[str]:
    """
    Грубое, но стабильное разбиение сценария на "сцены".

    Не пытаемся идеально понимать структуру.
    Идея:
    - сначала режем по двойным переводам строки
    - собираем блоки в сцены длиной от min_scene_chars до max_scene_chars
    """

    if not text:
        return []

    # нормализуем переносы строк
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # делим по пустым строкам
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    scenes: List[str] = []
    buf: List[str] = []
    cur_len = 0

    for block in blocks:
        block_len = len(block)
        # если текущая сцена уже достаточно большая — закрываем её
        if cur_len >= min_scene_chars and (cur_len + block_len > max_scene_chars):
            scenes.append("\n\n".join(buf))
            buf = []
            cur_len = 0

        buf.append(block)
        cur_len += block_len

    if buf:
        scenes.append("\n\n".join(buf))

    # если всё равно получилась одна гигантская сцена — режем по длине
    if len(scenes) == 1 and len(scenes[0]) > max_scene_chars * 2:
        big = scenes[0]
        scenes = []
        for i in range(0, len(big), max_scene_chars):
            chunk = big[i : i + max_scene_chars]
            if chunk.strip():
                scenes.append(chunk.strip())

    return scenes
