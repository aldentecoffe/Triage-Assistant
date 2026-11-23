"""Нормализация текста и хеш описания.

Общий модуль для генератора, очистки, split и тестов: хеш должен считаться
одинаково на всех стадиях, иначе очистка не найдёт дубликат.
"""

from __future__ import annotations

import hashlib

# Минимальная длина описания, при которой карточка годится для обучения.
MIN_TOKEN_COUNT = 5


def normalise(text: str) -> str:
    """Регистр и повторные пробелы; пунктуация сохраняется."""
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    """SHA-256 нормализованного описания: поиск дублей и проверка разделения."""
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()
