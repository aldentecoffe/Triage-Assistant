"""Проверки демонстрационного расчёта согласия.

Демонстрация не заменяет пилот: тесты держат только то, что прогон
воспроизводим и что изменения второго прохода действительно детерминированы.
"""

from __future__ import annotations

import pandas as pd
import pytest

import agreement
import agreement_demo as demo


@pytest.fixture
def source(tmp_path) -> str:
    rows = []
    for number in range(1, 31):
        rows.append(
            {
                "incident_id": f"I{number:03d}",
                "description": f"Описание {number}",
                "severity": ["P1", "P2", "P3", "P4"][number % 4],
                "service": ["billing", "bss", "platform", "database", "transport"][number % 5],
                "root_cause": ["change", "capacity", "hardware", "software"][number % 4],
            }
        )
    path = tmp_path / "validation.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return str(path)


def test_demo_sheets_are_deterministic_and_complete(source: str) -> None:
    first, second, key, gold = demo.build_sheets(source, 30)

    assert len(first) == len(second) == len(gold) == 30
    assert len(key) == 60
    again = demo.build_sheets(source, 30)
    assert first.equals(again[0]) and second.equals(again[1])


def test_demo_changes_exactly_the_planned_rows(source: str) -> None:
    first, second, _, _ = demo.build_sheets(source, 30)
    changed = first.merge(second, on="card", suffixes=("_a", "_b"))
    severity_diff = int((changed["severity_a"] != changed["severity_b"]).sum())
    service_diff = int((changed["service_a"] != changed["service_b"]).sum())
    cause_diff = int((changed["root_cause_a"] != changed["root_cause_b"]).sum())

    assert severity_diff == len(range(0, 30, demo.SEVERITY_STEP))
    assert service_diff == len(range(0, 30, demo.SERVICE_STEP))
    assert cause_diff == len(range(0, 30, demo.CAUSE_STEP))


def test_demo_report_uses_the_same_code_as_the_pilot(source: str) -> None:
    first, second, key, gold = demo.build_sheets(source, 30)
    text = agreement.report(agreement.combine(first, second, key, gold))

    assert "### Согласие первого и второго прохода" in text
    assert "### Сверка с метками набора" in text
    assert "Квадратичная каппа" in text
