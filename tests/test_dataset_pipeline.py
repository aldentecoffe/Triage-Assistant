"""Тесты пайплайна данных лабораторной 4.

Проверяют детерминированность генератора, порядок очистки, отсутствие
пересечений между split и полноту схемы.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_incidents import build_rows  # noqa: E402
from incident_taxonomy import (  # noqa: E402
    CASCADES,
    DEFECTS,
    REQUIRED_COLUMNS,
    SCENARIOS,
    SERVICES,
)
from prepare_dataset import clean  # noqa: E402
from split_dataset import assign, find_violations  # noqa: E402
from text_utils import text_hash  # noqa: E402

PII_COLUMNS = {
    "email",
    "phone",
    "fio",
    "full_name",
    "login",
    "passport",
    "address",
    "ip_address",
}


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.DataFrame(build_rows(total=400))


@pytest.fixture(scope="module")
def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    frame, _ = clean(raw)
    return frame


def test_generator_is_deterministic() -> None:
    first = pd.DataFrame(build_rows(total=120))
    second = pd.DataFrame(build_rows(total=120))
    assert first.equals(second)


def test_generator_adds_known_number_of_defects(raw: pd.DataFrame) -> None:
    expected = 400 + sum(DEFECTS.values())
    assert len(raw) == expected
    assert int(raw["rights_status"].eq("unknown").sum()) == DEFECTS["rights_unknown"]
    empty = raw["description"].str.strip().eq("")
    assert int(empty.sum()) == DEFECTS["empty_description"]

    # Копии без подтверждённого допуска тоже совпадают по тексту с оригиналами.
    # Их снимает первая проверка, поэтому дубликаты считаем после неё.
    admissible = raw[
        raw["rights_status"].eq("demo_allowed") & raw["description"].str.strip().ne("")
    ]
    duplicates = admissible["text_hash"].duplicated(keep="first")
    assert int(duplicates.sum()) == DEFECTS["duplicate_text"]


def test_cleaning_removes_exactly_the_defects(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    _, log = clean(raw)
    assert len(cleaned) == len(raw) - sum(DEFECTS.values())
    assert log["reason"].value_counts().to_dict() == {
        "duplicate_text": DEFECTS["duplicate_text"],
        "empty_or_short_description": DEFECTS["empty_description"],
        "rights_unknown": DEFECTS["rights_unknown"],
    }


def test_cleaned_frame_has_no_defects(cleaned: pd.DataFrame) -> None:
    assert cleaned["rights_status"].eq("demo_allowed").all()
    assert cleaned["description"].str.strip().str.split().str.len().ge(5).all()
    assert not cleaned["text_hash"].duplicated().any()


def test_cleaning_order_is_fixed() -> None:
    rows = [
        {
            "incident_id": "I000001",
            "description": "падает сессия на узле транзита из-за петли",
            "rights_status": "demo_allowed",
        },
        {
            "incident_id": "I000002",
            "description": "падает сессия на узле транзита из-за петли",
            "rights_status": "demo_allowed",
        },
        {"incident_id": "I000003", "description": "", "rights_status": "unknown"},
        {"incident_id": "I000004", "description": "", "rights_status": "demo_allowed"},
        {"incident_id": "I000005", "description": "   ", "rights_status": "demo_allowed"},
    ]
    frame = pd.DataFrame(rows)
    frame["text_hash"] = frame["description"].map(text_hash)

    kept, log = clean(frame)
    reasons = log.set_index("incident_id")["reason"].to_dict()

    assert list(kept["incident_id"]) == ["I000001"]
    assert reasons["I000002"] == "duplicate_text"
    assert reasons["I000003"] == "rights_unknown"
    assert reasons["I000004"] == "empty_or_short_description"
    assert reasons["I000005"] == "empty_or_short_description"


def test_split_has_no_leaks(cleaned: pd.DataFrame) -> None:
    frame = assign(cleaned)
    assert find_violations(frame) == []


def test_every_group_belongs_to_one_split(cleaned: pd.DataFrame) -> None:
    frame = assign(cleaned)
    for key in ("scenario_id", "event_id", "text_hash"):
        assert frame.groupby(key)["split"].nunique().max() == 1


def test_cascades_share_the_same_split() -> None:
    split_by_scenario = {scenario["id"]: scenario["split"] for scenario in SCENARIOS}
    for left, right in CASCADES:
        assert split_by_scenario[left] == split_by_scenario[right]


def test_every_split_covers_all_services(cleaned: pd.DataFrame) -> None:
    frame = assign(cleaned)
    for split in ("train", "validation", "test"):
        part = frame[frame["split"] == split]
        assert set(part["service"]) == set(SERVICES)


def test_schema_is_complete_and_free_of_personal_fields(cleaned: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    assert missing == []
    assert PII_COLUMNS.isdisjoint(cleaned.columns)


def test_text_hash_matches_description(cleaned: pd.DataFrame) -> None:
    assert cleaned["description"].map(text_hash).equals(cleaned["text_hash"])


def test_language_mix_is_documented(cleaned: pd.DataFrame) -> None:
    assert set(cleaned["language"]) == {"ru", "en"}
    ru_share = cleaned["language"].eq("ru").mean()
    assert 0.7 <= ru_share <= 0.9
