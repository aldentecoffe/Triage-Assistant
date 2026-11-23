"""Общие фикстуры тестов лабораторных 4–6.

Набор размера по умолчанию генерируется один раз за прогон: он нужен
нескольким модулям, а генерация и очистка не бесплатны. Модули ниже
только нарезают из него нужные куски.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_incidents import build_rows  # noqa: E402
from prepare_dataset import clean  # noqa: E402
from split_dataset import assign  # noqa: E402

SPLIT_ORDER = ("train", "validation", "test")


@pytest.fixture(scope="session")
def default_frame() -> pd.DataFrame:
    """Очищенный набор размера по умолчанию (6 000 записей) с назначенными split."""
    cleaned, _ = clean(pd.DataFrame(build_rows()))
    return assign(cleaned)


@pytest.fixture(scope="session")
def default_splits(default_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Три выборки набора по умолчанию, разделённые по полю split."""
    return {
        split: default_frame[default_frame["split"] == split].reset_index(drop=True)
        for split in SPLIT_ORDER
    }
