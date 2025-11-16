from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml.age_inference import AgeRatingService
from ml.lexicons import (
    VIOLENCE_PATTERNS,
    EROTICA_MILD_PATTERNS,
    EROTICA_HARD_PATTERNS,
    PROFANITY_PATTERNS,
    SUBSTANCES_MILD_PATTERNS,
    SUBSTANCES_HARD_PATTERNS,
    SCARY_PATTERNS,
)
from data_prep.pdf_to_scenes import extract_text_from_pdf_bytes, split_into_scenes
from data_prep.doc_to_scenes import extract_text_from_docx_bytes

app = FastAPI(
    title="Age Rating Service",
    version="1.0.0",
)

# CORS, чтобы фронт из браузера мог ходить к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализируем сервис один раз при старте
service = AgeRatingService(model_dir="models/age_model")


# -------------------- паттерны по категориям --------------------

CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "violence": VIOLENCE_PATTERNS,
    "erotica": EROTICA_MILD_PATTERNS + EROTICA_HARD_PATTERNS,
    "profanity": PROFANITY_PATTERNS,
    "substances": SUBSTANCES_MILD_PATTERNS + SUBSTANCES_HARD_PATTERNS,
    "scary": SCARY_PATTERNS,
}


def _collect_matches(text: str, patterns: List[str]) -> tuple[List[tuple[int, int]], List[str]]:
    """
    Возвращает:
      - список span'ов (start, end) для подсветки;
      - список уникальных найденных фрагментов (для вывода в JSON).
    """
    spans: List[tuple[int, int]] = []
    fragments: List[str] = []
    seen: set[str] = set()

    for p in patterns:
        try:
            it = re.finditer(p, text, flags=re.IGNORECASE | re.MULTILINE)
        except re.error:
            # На всякий случай, если регулярка сломана
            continue

        for m in it:
            s, e = m.span()
            spans.append((s, e))
            frag = m.group(0)
            norm = frag.lower()
            if norm not in seen:
                seen.add(norm)
                fragments.append(frag)

    return spans, fragments


def _merge_spans(spans: List[tuple[int, int]]) -> List[tuple[int, int]]:
    """
    Склеиваем пересекающиеся / вложенные интервалы, чтобы не делать вложенные <mark>.
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: x[0])
    merged: List[tuple[int, int]] = []
    cur_start, cur_end = spans[0]

    for s, e in spans[1:]:
        if s <= cur_end:  # пересечение/стык
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e

    merged.append((cur_start, cur_end))
    return merged


def _highlight_text(text: str, spans: List[tuple[int, int]]) -> str:
    """
    Оборачиваем проблемные куски в <mark>...</mark>.
    """
    if not spans:
        return text

    spans = _merge_spans(spans)
    parts: List[str] = []
    last = 0

    for s, e in spans:
        if s < last:
            continue
        parts.append(text[last:s])
        parts.append("<mark>")
        parts.append(text[s:e])
        parts.append("</mark>")
        last = e

    parts.append(text[last:])
    return "".join(parts)


# --------- Генерация советов по исправлению ---------


def make_issue_suggestion(
    category: str,
    severity_index: int,
    text: str,
    fragments: Optional[List[str]] = None,
) -> str:
    """
    Возвращает текстовый совет, как смягчить/исправить сцену
    по конкретной категории и уровню тяжести.
    fragments — список конкретных найденных слов/фраз.
    """
    prefix = ""
    if fragments:
        show = fragments[:5]
        quoted = ", ".join(f"«{w}»" for w in show)
        prefix = f"Проблемные слова/фразы: {quoted}. "

    # Нецензурная лексика / оскорбления
    if category == "profanity":
        return (
            prefix
            + "Замените грубые и нецензурные слова на нейтральные выражения, "
              "уберите прямые оскорбления. Конфликт можно передать через интонацию, "
              "реакцию персонажей или мягкие формулировки без мата."
        )

    # Эротика / сексуальный контент
    if category == "erotica":
        if severity_index <= 1:
            return (
                prefix
                + "Сократите или уберите намёки сексуального характера: не описывайте подробности "
                  "внешности и поведения с сексуальным подтекстом, избегайте двусмысленных шуток. "
                  "Оставьте только общий контекст без акцента на сексуальности."
            )
        else:
            return (
                prefix
                + "Исключите сцены и описания сексуальных действий, откровенные намёки и подробности. "
                  "Если сцена важна для сюжета, передайте её максимально обобщённо, "
                  "без прямого описания происходящего."
            )

    # Насилие
    if category == "violence":
        if severity_index == 1:
            return (
                prefix
                + "Смягчите конфликты: уберите прямые описания ударов, избиений и угроз расправой. "
                  "Можно оставить спор или ссору, но без физического насилия и жёстких формулировок."
            )
        else:
            return (
                prefix
                + "Уберите или сократите сцены насилия и угроз, не описывайте детали "
                  "телесных повреждений, крови или смерти. Сконцентрируйтесь на последствиях "
                  "и эмоциях персонажей без реалистичных жестоких сцен."
            )

    # Вещества (алкоголь, наркотики и т.п.)
    if category == "substances":
        if severity_index == 1:
            return (
                prefix
                + "Минимизируйте упоминания алкоголя и других веществ: не показывайте употребление "
                  "как норму или что-то забавное. Если сцена важна, упомяните это вскользь, "
                  "без акцента и подробностей."
            )
        else:
            return (
                prefix
                + "Уберите сцены употребления наркотиков и злоупотребления алкоголем, особенно если "
                  "это показано подробно или в позитивном ключе. Можно лишь упомянуть факт "
                  "проблемы без демонстрации процесса."
            )

    # Страшные моменты / хоррор
    if category == "scary":
        return (
            prefix
            + "Сделайте страшные элементы менее реалистичными: уменьшите количество описаний угроз, "
              "боли и смерти, уберите натуралистичные детали. Оставьте сказочный или мистический "
              "элемент в более мягком, приключенческом или комичном ключе."
        )

    # Дефолтный совет
    return (
        prefix
        + "Смягчите формулировки в этой сцене: уберите жёсткие слова и подробные описания, "
          "оставив общий смысл без деталей, которые повышают возрастной рейтинг."
    )


def enrich_problem_scenes(
    problem_scenes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    На вход: problem_scenes из ядра (AgeRatingService),
    На выход: тот же список, но:
      - у каждого issue:
          * suggestion
          * offending_fragments
      - у каждой сцены:
          * scene_suggestion (общий совет)
          * highlighted_text_snippet (с <mark> для всех проблемных слов)
    """
    enriched: List[Dict[str, Any]] = []

    for scene in problem_scenes:
        text = scene.get("text_snippet", "")
        issues = scene.get("issues", []) or []

        scene_spans_all: List[tuple[int, int]] = []
        scene_suggestion_parts: List[str] = []
        new_issues: List[Dict[str, Any]] = []

        for issue in issues:
            cat = issue.get("category")
            sev_idx = int(issue.get("severity_index", 0))

            patterns = CATEGORY_PATTERNS.get(cat, [])
            spans: List[tuple[int, int]]
            fragments: List[str]

            if patterns:
                spans, fragments = _collect_matches(text, patterns)
            else:
                spans, fragments = [], []

            scene_spans_all.extend(spans)

            suggestion = make_issue_suggestion(cat, sev_idx, text, fragments)
            scene_suggestion_parts.append(suggestion)

            new_issue = dict(issue)
            new_issue["offending_fragments"] = fragments
            new_issue["suggestion"] = suggestion
            new_issues.append(new_issue)

        scene_copy = dict(scene)
        scene_copy["issues"] = new_issues

        # общий совет по сцене
        if scene_suggestion_parts:
            scene_copy["scene_suggestion"] = " ".join(scene_suggestion_parts)

        # подсвеченный текст
        highlighted = _highlight_text(text, scene_spans_all)
        scene_copy["highlighted_text_snippet"] = highlighted

        enriched.append(scene_copy)

    return enriched


# --------- разбор файла в сцены ---------


def file_bytes_to_scenes(file_bytes: bytes, filename: str) -> List[str]:
    """
    Принимает байты файла и имя, возвращает список сцен (строк),
    готовых для AgeRatingService.analyze_script.
    """
    name_lower = filename.lower()
    _, ext = os.path.splitext(name_lower)
    ext = ext.lstrip(".")

    text: str

    if ext == "pdf":
        text = extract_text_from_pdf_bytes(file_bytes)
    elif ext == "docx":
        text = extract_text_from_docx_bytes(file_bytes)
    elif ext == "txt":
        # простой вариант для txt
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Формат файла .{ext} не поддерживается. Ожидается pdf / docx / txt.",
        )

    scenes = split_into_scenes(text)

    # на всякий случай: если разбиение дало пусто, берём весь текст одной сценой
    if not scenes and text.strip():
        scenes = [text]

    return scenes


# --------- Основной эндпоинт ---------


@app.post("/analyze_file")
async def analyze_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Главная ручка:
    - принимает файл сценария (pdf / docx / txt),
    - конвертит в список сцен,
    - гоняет через AgeRatingService,
    - добавляет советы и подсветку проблемных слов,
    - возвращает итоговый JSON.
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_file"

    scenes = file_bytes_to_scenes(file_bytes, filename)

    # ВАЖНО: сюда передаём уже список сцен, как ждёт AgeRatingService
    core_result: Dict[str, Any] = service.analyze_script(scenes, filename)

    # обогащаем problem_scenes подсветкой и советами
    raw_problem_scenes = core_result.get("problem_scenes", []) or []
    enriched_problem_scenes = enrich_problem_scenes(raw_problem_scenes)
    core_result["problem_scenes"] = enriched_problem_scenes

    return core_result
