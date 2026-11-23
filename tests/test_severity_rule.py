"""Правило критичности версии 1.1: метка выводится из формулировки последствия.

`IMPACT_SEVERITY` заменил случайное разыгрывание severity: карточка получает
класс того последствия, чья фраза есть в описании. Тесты держат полноту
словаря, само правило на сгенерированном наборе и состав выборок.
"""

from __future__ import annotations

import pandas as pd
import pytest

from generate_incidents import build_rows, minimum_total
from incident_taxonomy import (
    CASCADES,
    DATASET_VERSION,
    GENERATOR_VERSION,
    IMPACT_SEVERITY,
    SCENARIOS,
    SEVERITIES,
    SPLIT_SHARES,
)
from prepare_dataset import clean
from split_dataset import assign

SCENARIO_BY_ID = {scenario["id"]: scenario for scenario in SCENARIOS}


def severity_conflicts(frame: pd.DataFrame) -> list[str]:
    """Карточки, где severity расходится с классом последствия из собственного текста."""
    conflicts: list[str] = []
    for _, row in frame.iterrows():
        scenario = SCENARIO_BY_ID[row["scenario_id"]]
        marks = IMPACT_SEVERITY[row["scenario_id"]]
        text = row["description"].lower()
        if row["language"] == "en":
            hits = [marks["en"]] if scenario["impact_en"].lower() in text else []
        else:
            hits = [
                label
                for phrase, label in zip(scenario["impacts_ru"], marks["ru"])
                if phrase.lower() in text
            ]
        if hits != [row["severity"]]:
            found = hits[0] if len(hits) == 1 else f"{len(hits)} последствий"
            conflicts.append(f"{row['incident_id']}: severity {row['severity']}, в тексте {found}")
    return conflicts


def test_impact_severity_covers_exactly_the_catalogue() -> None:
    """Словарь описывает все 28 сценариев и не содержит лишних ключей."""
    assert set(IMPACT_SEVERITY) == set(SCENARIO_BY_ID)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[item["id"] for item in SCENARIOS])
def test_every_scenario_maps_all_its_impacts(scenario: dict) -> None:
    """У сценария два русских последствия и одно английское, все метки из P1–P4."""
    marks = IMPACT_SEVERITY[scenario["id"]]

    assert len(marks["ru"]) == len(scenario["impacts_ru"]) == 2
    assert isinstance(marks["en"], str)
    assert set(marks["ru"]) <= set(SEVERITIES)
    assert marks["en"] in SEVERITIES


def test_every_card_severity_follows_its_impact(default_frame: pd.DataFrame) -> None:
    """На наборе по умолчанию метка каждой карточки воспроизводится по тексту."""
    assert set(default_frame["severity"]) <= set(SEVERITIES)
    assert severity_conflicts(default_frame) == []


def test_language_column_has_only_known_values(default_frame: pd.DataFrame) -> None:
    """Правило опирается на язык описания, поэтому других значений быть не должно."""
    assert set(default_frame["language"]) == {"ru", "en"}


def test_rule_holds_on_the_smallest_dataset() -> None:
    """Правило не зависит от объёма: минимальный набор из 34 записей тоже сходится."""
    raw = pd.DataFrame(build_rows(total=minimum_total()))
    frame = assign(clean(raw)[0])

    assert severity_conflicts(frame) == []


def test_records_are_marked_with_version_1_1(default_frame: pd.DataFrame) -> None:
    """Версии генератора и набора подняты до 1.1 и проставлены в каждой записи."""
    assert (GENERATOR_VERSION, DATASET_VERSION) == ("1.1", "1.1")
    assert set(default_frame["generator_version"]) == {GENERATOR_VERSION}
    assert set(default_frame["dataset_version"]) == {DATASET_VERSION}
    assert default_frame["source"].str.contains(f"@{GENERATOR_VERSION}", regex=False).all()


def test_every_split_contains_all_four_severities(default_frame: pd.DataFrame) -> None:
    """После перераспределения сценариев каждый класс критичности есть во всех выборках."""
    for split in SPLIT_SHARES:
        present = set(default_frame.loc[default_frame["split"] == split, "severity"])
        assert present == set(SEVERITIES), split


def test_cascades_stay_inside_one_split(default_frame: pd.DataFrame) -> None:
    """Обе половины каскадной пары попадают в одну выборку, иначе метрика врёт."""
    split_by_scenario = default_frame.groupby("scenario_id")["split"].first()

    for left, right in CASCADES:
        assert split_by_scenario[left] == split_by_scenario[right], (left, right)


def test_p1_share_stays_between_four_and_ten_percent(default_frame: pd.DataFrame) -> None:
    """Доля P1 держится в коридоре 4–10 %: защита от регресса правила в любую сторону."""
    share = default_frame["severity"].eq("P1").mean()

    assert 0.04 <= share <= 0.10, f"доля P1 {share:.3f}"
