"""Тесты диагностики leave-one-scenario-out (ЛР 10, проверка устойчивости).

Проверяется механика фолдов и то, что числа в документе берутся из отчёта:
сценарий не попадает одновременно в индекс и в запросы, метрики фолда считаются
на мини-корпусе, пул считается по полной таксономии, а документ повторяет
значения из JSON. Полный прогон на реальном наборе в тестах не повторяется —
его делает команда `python src/scenario_cv.py run`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scenario_cv

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "docs" / "diagnostics" / "results" / "scenario_cv.json"
DOCUMENT = REPO_ROOT / "docs" / "diagnostics" / "scenario_cv.md"


def mini_frame() -> pd.DataFrame:
    """Три сценария по шесть карточек: разные услуги, причины и словари."""
    rows = []
    texts = {
        "SC-A-01": "падает bgp сессия пиринг магистраль задержка префиксы",
        "SC-B-01": "биллинг очередь списаний растёт задержка обработки заказов",
        "SC-C-01": "ims регистрация абонентов отказы голосовой сервис",
    }
    meta = {
        "SC-A-01": ("transport", "network", "P2"),
        "SC-B-01": ("billing", "capacity", "P3"),
        "SC-C-01": ("ims", "software", "P1"),
    }
    for scenario, text in texts.items():
        service, cause, severity = meta[scenario]
        for index in range(6):
            rows.append(
                {
                    "incident_id": f"I{scenario[-4:]}{index:02d}",
                    "scenario_id": scenario,
                    "event_id": f"E{scenario[-4:]}{index // 2}",
                    "description": f"{text} ci-{index:04d}",
                    "text_hash": f"{scenario}-{index}",
                    "severity": severity,
                    "service": service,
                    "root_cause": cause,
                    "dataset_version": "1.1",
                }
            )
    return pd.DataFrame(rows)


# --- фолды ----------------------------------------------------------------


def test_fold_scenarios_sorted_and_unique():
    assert scenario_cv.fold_scenarios(mini_frame()) == ["SC-A-01", "SC-B-01", "SC-C-01"]


def test_fold_scenarios_requires_two():
    frame = mini_frame()
    with pytest.raises(ValueError, match="два сценария"):
        scenario_cv.fold_scenarios(frame[frame["scenario_id"] == "SC-A-01"])


def test_split_fold_has_no_overlap():
    frame = mini_frame()
    index, queries = scenario_cv.split_fold(frame, "SC-B-01")
    assert set(index["scenario_id"]) == {"SC-A-01", "SC-C-01"}
    assert set(queries["scenario_id"]) == {"SC-B-01"}
    assert not set(index["text_hash"]) & set(queries["text_hash"])


def test_split_fold_rejects_duplicate_texts():
    frame = mini_frame()
    frame.loc[frame["scenario_id"] == "SC-A-01", "text_hash"] = "same"
    frame.loc[frame["scenario_id"] == "SC-B-01", "text_hash"] = "same"
    with pytest.raises(ValueError, match="общих текстов"):
        scenario_cv.split_fold(frame, "SC-B-01")


# --- поиск ----------------------------------------------------------------


def test_ranked_top_orders_by_score_then_id():
    scores = np.array([0.2, 0.9, 0.9, 0.0])
    ids = np.array(["I2", "I1", "I0", "I3"])
    assert list(scenario_cv.ranked_top(scores, ids, 2)) == [2, 1]


def test_ranked_top_drops_zero_scores():
    assert scenario_cv.ranked_top(np.array([0.0, 0.0]), np.array(["a", "b"]), 3).size == 0


def test_ranked_top_zero_k_is_empty():
    assert scenario_cv.ranked_top(np.array([1.0]), np.array(["a"]), 0).size == 0


# --- метрики фолда --------------------------------------------------------


def test_evaluate_fold_reports_all_metrics():
    frame = mini_frame()
    index, queries = scenario_cv.split_fold(frame, "SC-B-01")
    summary, record = scenario_cv.evaluate_fold(index, queries)
    assert summary["scenario_id"] == "SC-B-01"
    assert summary["cards"] == 6
    for metric in scenario_cv.FOLD_METRICS:
        assert metric in summary["metrics"]
    assert 0.0 <= summary["metrics"]["ndcg@3"] <= 1.0
    assert set(record.columns) >= {"true_service", "vote_service", "clf_service", "rule_severity"}


def test_evaluate_fold_counts_rule_coverage_zero_without_known_phrases():
    frame = mini_frame()
    index, queries = scenario_cv.split_fold(frame, "SC-A-01")
    summary, _ = scenario_cv.evaluate_fold(index, queries)
    assert summary["metrics"]["rule_coverage"] == 0.0
    assert summary["metrics"]["rule_accuracy"] is None


def test_evaluate_fold_no_leakage_by_scenario():
    frame = mini_frame()
    index, queries = scenario_cv.split_fold(frame, "SC-C-01")
    summary, _ = scenario_cv.evaluate_fold(index, queries)
    assert summary["index_cards"] == 12


# --- пул, сводка, сравнение ----------------------------------------------


def test_pooled_metrics_perfect_predictions():
    services = ["billing", "bss", "database", "external", "ims", "platform", "transport"]
    causes = ["capacity", "change", "external_dep", "hardware", "network", "software", "software"]
    severities = ["P1", "P2", "P3", "P4", "P1", "P2", "P3"]
    frame = pd.DataFrame(
        {
            "true_service": services,
            "vote_service": services,
            "clf_service": services,
            "true_root_cause": causes,
            "vote_root_cause": causes,
            "clf_root_cause": causes,
            "true_severity": severities,
            "vote_severity": severities,
            "clf_severity": severities,
        }
    )
    pooled = scenario_cv.pooled_metrics(frame)
    assert pooled["vote_service_f1"] == pytest.approx(1.0, abs=1e-9)
    assert pooled["clf_cause_f1"] == pytest.approx(1.0, abs=1e-9)
    assert pooled["clf_p1_recall"] == 1.0


def test_pooled_metrics_uses_full_taxonomy():
    """Два верных класса из семи дают macro-F1 около 2/7, а не единицу."""
    frame = pd.DataFrame(
        {
            "true_service": ["ims", "bss"],
            "vote_service": ["ims", "bss"],
            "clf_service": ["ims", "bss"],
            "true_root_cause": ["software", "change"],
            "vote_root_cause": ["software", "change"],
            "clf_root_cause": ["software", "change"],
            "true_severity": ["P1", "P3"],
            "vote_severity": ["P1", "P3"],
            "clf_severity": ["P1", "P3"],
        }
    )
    pooled = scenario_cv.pooled_metrics(frame)
    assert pooled["vote_service_f1"] == pytest.approx(2 / 7, abs=1e-6)


def test_pooled_metrics_penalises_missing_class():
    frame = pd.DataFrame(
        {
            "true_service": ["ims", "bss"],
            "vote_service": ["ims", "ims"],
            "clf_service": ["ims", "ims"],
            "true_root_cause": ["software", "change"],
            "vote_root_cause": ["software", "software"],
            "clf_root_cause": ["software", "software"],
            "true_severity": ["P1", "P3"],
            "vote_severity": ["P1", "P3"],
            "clf_severity": ["P1", "P3"],
        }
    )
    pooled = scenario_cv.pooled_metrics(frame)
    assert pooled["vote_service_f1"] < 1.0
    assert pooled["clf_cause_f1"] < 1.0


def test_summarise_reports_bounds_and_skips_none():
    folds = [
        {"metrics": {metric: 0.2 for metric in scenario_cv.FOLD_METRICS}},
        {"metrics": {metric: 0.4 for metric in scenario_cv.FOLD_METRICS}},
    ]
    folds[1]["metrics"]["rule_accuracy"] = None
    summary = scenario_cv.summarise(folds)
    assert summary["ndcg@3"]["mean"] == pytest.approx(0.3)
    assert summary["ndcg@3"]["min"] == 0.2
    assert summary["ndcg@3"]["max"] == 0.4
    assert summary["rule_accuracy"]["folds"] == 1


def test_comparison_block_delta_sign():
    pooled = {"vote_service_f1": 0.1, "clf_cause_f1": 0.4}
    reference = {"vote_service_f1": 0.3, "clf_cause_f1": 0.2}
    block = scenario_cv.comparison_block(pooled, reference)
    assert block["vote_service_f1"]["delta"] == pytest.approx(-0.2)
    assert block["clf_cause_f1"]["delta"] == pytest.approx(0.2)


# --- документ -------------------------------------------------------------


@pytest.fixture
def mini_report(monkeypatch):
    """Полный отчёт на мини-корпусе: фолды, сводка и документ без 13 секунд счёта."""
    frame = mini_frame()
    monkeypatch.setattr(
        scenario_cv,
        "load_selected_splits",
        lambda splits: {"train": frame, "validation": frame.tail(6)},
    )
    monkeypatch.setattr(scenario_cv, "input_hashes", lambda splits: {})
    monkeypatch.setattr(scenario_cv, "LAB10_SUMMARY", Path("нет.json"))
    monkeypatch.setattr(scenario_cv, "LAB9_SUMMARY", Path("нет.json"))
    monkeypatch.setattr(scenario_cv, "LAB7_METRICS", Path("нет.json"))
    return scenario_cv.build_report()


def test_render_markdown_mentions_every_fold(mini_report):
    text = scenario_cv.render_markdown(mini_report)
    for fold in mini_report["per_fold"]:
        assert fold["scenario_id"] in text
    assert "leave-one-scenario-out" in text
    assert "nan" not in text.lower()


def test_published_report_shape():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["splits_used"] == ["train"]
    assert payload["test_used"] is False
    assert payload["folds"] == len(payload["per_fold"]) == 14
    assert payload["cards"] == 3600
    assert set(payload["fold_summary"]) == set(scenario_cv.FOLD_METRICS)
    assert payload["proximity"]["folds_min"] <= payload["proximity"]["folds_mean"] <= payload["proximity"]["folds_max"]


def test_published_document_quotes_json():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    text = DOCUMENT.read_text(encoding="utf-8")
    for fold in payload["per_fold"]:
        assert fold["scenario_id"] in text
    service = f"{payload['pooled']['vote_service_f1']:.3f}".replace(".", ",")
    assert service in text
    proximity = f"{payload['proximity']['validation']:.3f}".replace(".", ",")
    assert proximity in text


def test_published_rule_is_stable_across_folds():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    for fold in payload["per_fold"]:
        assert fold["metrics"]["rule_coverage"] == 1.0
        assert fold["metrics"]["rule_accuracy"] == 1.0


def test_run_writes_report_and_document(mini_report, tmp_path, monkeypatch):
    monkeypatch.setattr(
        scenario_cv, "build_report", lambda: mini_report
    )
    code = scenario_cv.run(tmp_path / "results", tmp_path / "docs")
    assert code == 0
    payload = json.loads((tmp_path / "results" / "scenario_cv.json").read_text(encoding="utf-8"))
    assert payload["folds"] == 3
    assert (tmp_path / "docs" / "scenario_cv.md").exists()
