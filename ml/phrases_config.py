# ml/phrases_config.py
# -*- coding: utf-8 -*-
"""
Прослойка между регулярками в lexicons.py и сервисом age_inference.

Здесь:
- CATEGORIES                     — список всех категорий
- detect_categories_for_scene    — по тексту сцены возвращает категории + severity
- age_from_categories_for_scene  — по категориям возвращает возраст сцены (6/12/16/18)
"""

import re
from typing import Dict, Any

from ml.lexicons import (
    VIOLENCE_PATTERNS,
    EROTICA_MILD_PATTERNS,
    EROTICA_HARD_PATTERNS,
    PROFANITY_PATTERNS,
    SUBSTANCES_MILD_PATTERNS,
    SUBSTANCES_HARD_PATTERNS,
    SCARY_PATTERNS,
)

# те же самые категории, что ждёт age_inference
CATEGORIES = ["violence", "erotica", "profanity", "substances", "scary"]

SEVERITY_ORDER = ["none", "mild", "moderate", "severe"]
SEVERITY_TO_INDEX = {name: idx for idx, name in enumerate(SEVERITY_ORDER)}


def _severity_none() -> Dict[str, Any]:
    return {
        "severity": "none",
        "severity_index": 0,
        "confidence": 1.0,
    }


def _make_severity(severity: str, confidence: float = 1.0) -> Dict[str, Any]:
    return {
        "severity": severity,
        "severity_index": SEVERITY_TO_INDEX[severity],
        "confidence": confidence,
    }


def _count_matches(patterns, text: str) -> int:
    cnt = 0
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE | re.MULTILINE):
            cnt += 1
    return cnt


def _severity_by_count(count: int, hard: bool = False) -> Dict[str, Any]:
    """
    Грубая эвристика:
    - нет совпадений -> none
    - если hard=True и count>0 -> severe
    - иначе: 1 -> mild, 2-3 -> moderate, 4+ -> severe
    """
    if count <= 0:
        return _severity_none()

    if hard:
        return _make_severity("severe")

    if count == 1:
        return _make_severity("mild")
    if 2 <= count <= 3:
        return _make_severity("moderate")
    return _make_severity("severe")


def detect_categories_for_scene(text: str) -> Dict[str, Dict[str, Any]]:
    """
    На входе текст сцены, на выходе:
    {
      "violence":  {"severity": "mild"/..., "severity_index": int, "confidence": float},
      "erotica":   {...},
      ...
    }
    Все слова/фразы берутся только из lexicons.py.
    """
    # Насилие
    violence_cnt = _count_matches(VIOLENCE_PATTERNS, text)
    violence = _severity_by_count(violence_cnt, hard=False)

    # Эротика: мягкая/жёсткая
    erotica_mild_cnt = _count_matches(EROTICA_MILD_PATTERNS, text)
    erotica_hard_cnt = _count_matches(EROTICA_HARD_PATTERNS, text)
    if erotica_hard_cnt > 0:
        erotica = _make_severity("severe")
    elif erotica_mild_cnt > 0:
        erotica = _make_severity("mild")
    else:
        erotica = _severity_none()

    # Ненормативка — пока считаем всё как "mild"
    profanity_cnt = _count_matches(PROFANITY_PATTERNS, text)
    if profanity_cnt > 0:
        profanity = _make_severity("mild")
    else:
        profanity = _severity_none()

    # Вещества: отдельно лёгкие/тяжёлые
    subst_mild_cnt = _count_matches(SUBSTANCES_MILD_PATTERNS, text)
    subst_hard_cnt = _count_matches(SUBSTANCES_HARD_PATTERNS, text)
    if subst_hard_cnt > 0:
        substances = _make_severity("severe")
    elif subst_mild_cnt > 0:
        substances = _make_severity("mild")
    else:
        substances = _severity_none()

    # Страшности
    scary_cnt = _count_matches(SCARY_PATTERNS, text)
    # Здесь тоже: 1-2 -> mild, 3-4 -> moderate, 5+ -> severe
    scary = _severity_by_count(scary_cnt, hard=False)

    return {
        "violence": violence,
        "erotica": erotica,
        "profanity": profanity,
        "substances": substances,
        "scary": scary,
    }


def age_from_categories_for_scene(cats: Dict[str, Dict[str, Any]]) -> int:
    """
    Переводит категориальные severity в возраст сцены (6 / 12 / 16 / 18).
    Логика здесь полностью отделена от слов/регулярок.
    """
    age = 6

    v = cats["violence"]["severity"]
    e = cats["erotica"]["severity"]
    p = cats["profanity"]["severity"]
    s = cats["substances"]["severity"]
    sc = cats["scary"]["severity"]

    # Насилие
    if v == "mild":
        age = max(age, 12)
    elif v in ("moderate", "severe"):
        age = max(age, 16)

    # Эротика
    if e == "mild":
        age = max(age, 12)
    elif e in ("moderate", "severe"):
        age = max(age, 18)

    # Мат / грубые оскорбления
    if p in ("mild", "moderate"):
        age = max(age, 12)
    elif p == "severe":
        age = max(age, 16)

    # Алкоголь / наркотики
    if s == "mild":
        age = max(age, 12)
    elif s in ("moderate", "severe"):
        age = max(age, 16)

    # Страшные сцены
    if sc == "mild":
        # пусть лёгкий хоррор ещё укладывается в 6+ / 12+, не поднимаем выше 12
        age = max(age, 6)
    elif sc in ("moderate", "severe"):
        age = max(age, 12)

    return age
