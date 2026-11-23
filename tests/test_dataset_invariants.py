"""Инварианты каталога и набора размера по умолчанию.

Числа из docs/lab4 держатся на этих тестах: если каталог или генератор
изменятся, таблицы в Data Card перестанут соответствовать действительности,
и тест упадёт раньше, чем это заметит человек.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_incidents import UNIT_EN, build_rows, minimum_total  # noqa: E402
from incident_taxonomy import (  # noqa: E402
    CASCADES,
    DEFECTS,
    ROOT_CAUSES,
    SCENARIOS,
    SERVICES,
    SEVERITIES,
    SPLIT_SHARES,
)
from prepare_dataset import clean  # noqa: E402
from split_dataset import assign, find_violations  # noqa: E402


def test_every_scenario_has_a_known_split() -> None:
    unknown = {scenario["id"]: scenario["split"] for scenario in SCENARIOS}
    assert set(unknown.values()) <= set(SPLIT_SHARES)


def test_severity_weights_are_probabilities() -> None:
    for scenario in SCENARIOS:
        weights = [scenario["severity"][label] for label in SEVERITIES]
        assert sum(weights) == pytest.approx(1.0), scenario["id"]
        assert min(weights) >= 0, scenario["id"]


def test_every_metric_unit_can_be_translated() -> None:
    units = {unit for scenario in SCENARIOS for (_, _, unit, _) in scenario["metrics"]}
    assert units <= set(UNIT_EN)


def test_every_split_covers_all_services_in_the_catalogue() -> None:
    for split in SPLIT_SHARES:
        services = {s["service"] for s in SCENARIOS if s["split"] == split}
        assert services == set(SERVICES), split


def test_cascades_stay_inside_one_split() -> None:
    split_by_scenario = {scenario["id"]: scenario["split"] for scenario in SCENARIOS}
    for left, right in CASCADES:
        assert split_by_scenario[left] == split_by_scenario[right]


def test_root_causes_are_all_used() -> None:
    used = {scenario["root_cause"] for scenario in SCENARIOS}
    assert used == set(ROOT_CAUSES)


def test_default_dataset_matches_the_data_card() -> None:
    rows = build_rows()
    assert len(rows) == 6008

    frame = assign(clean(_as_frame(rows))[0])
    assert len(frame) == 6000
    assert frame["scenario_id"].nunique() == 28
    assert frame["event_id"].nunique() == 4110
    assert frame["split"].value_counts().to_dict() == {
        "train": 3600,
        "validation": 1200,
        "test": 1200,
    }
    assert find_violations(frame) == []


def test_default_dataset_removes_exactly_the_injected_defects() -> None:
    rows = build_rows()
    cleaned, log = clean(_as_frame(rows))

    assert len(rows) - len(cleaned) == sum(DEFECTS.values())
    assert log["reason"].value_counts().to_dict() == {
        "duplicate_text": DEFECTS["duplicate_text"],
        "empty_or_short_description": DEFECTS["empty_description"],
        "rights_unknown": DEFECTS["rights_unknown"],
    }


def test_generator_rejects_count_below_minimum() -> None:
    required = minimum_total()

    with pytest.raises(ValueError, match=str(required)):
        build_rows(total=required - 1)

    assert len(build_rows(total=required)) - sum(DEFECTS.values()) == required


def _as_frame(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows)
