"""Тесты ЛР 10: измерение источников меток и схема компонентов.

Проверяется то, на чём стоит архитектурное решение: правила голосования,
подсчёт F1 по полному списку классов, кривая «покрытие — качество», правило
выбора варианта, ворота проекта и то, что опубликованные документы не
расходятся с `results/check.json`.

Замеры на полном наборе в тестах не повторяются: они занимают секунды, а
повторяют их команда `python src/arch_check.py run`. Здесь проверяются
функции на маленьких примерах и согласованность опубликованных artefacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import arch_check

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "docs" / "lab10" / "results" / "check.json"
FIGURE = REPO_ROOT / "docs" / "lab10" / "figures" / "architecture_flow.png"
DOCS = [REPO_ROOT / "docs" / "lab10" / name for name in ("README.md", "architecture_matrix.md",
                                                        "architecture.md", "architecture_decision.md")]


def check_payload() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


# --- соседи и голосование -------------------------------------------------


def test_neighbour_order_drops_zero_pairs():
    order = arch_check.neighbour_order(np.array([0.0, 0.4, 0.0, 0.2]), 3)
    assert list(order) == [1, 3]


def test_neighbour_order_keeps_train_order_on_ties():
    order = arch_check.neighbour_order(np.array([0.5, 0.5, 0.5]), 2)
    assert list(order) == [0, 1]


def test_neighbour_order_handles_k_larger_than_rows():
    order = arch_check.neighbour_order(np.array([0.3, 0.1]), 10)
    assert len(order) == 2


def test_neighbour_order_all_zero_is_empty():
    assert arch_check.neighbour_order(np.zeros(4), 3).size == 0


def test_weighted_vote_sums_cosine_weights():
    label = arch_check.weighted_vote(
        np.array([0.6, 0.7]), np.array(["ims", "ims"]), ["ims", "bss"]
    )
    assert label == "ims"


def test_weighted_vote_wins_on_total_not_on_single_score():
    label = arch_check.weighted_vote(
        np.array([0.9, 0.5, 0.5]), np.array(["bss", "ims", "ims"]), ["ims", "bss"]
    )
    assert label == "ims"


def test_weighted_vote_breaks_tie_by_nearest_neighbour():
    label = arch_check.weighted_vote(
        np.array([0.5, 0.1, 0.3, 0.3]),
        np.array(["ims", "ims", "bss", "bss"]),
        ["ims", "bss"],
    )
    assert label == "ims"


def test_weighted_vote_returns_none_without_positive_scores():
    assert arch_check.weighted_vote(np.zeros(2), np.array(["ims", "bss"]), ["ims", "bss"]) is None


def test_weighted_vote_ignores_unknown_labels():
    label = arch_check.weighted_vote(np.array([0.9, 0.4]), np.array(["other", "ims"]), ["ims"])
    assert label == "ims"


def test_vote_support_counts_share_of_winner():
    from scipy.sparse import csr_matrix

    frames = {
        "train": pd.DataFrame(
            {
                "description": ["a", "b"],
                "severity": ["P1", "P2"],
                "service": ["ims", "bss"],
                "root_cause": ["software", "change"],
            }
        )
    }
    validation_matrix = csr_matrix(np.array([[1.0, 0.0], [0.6, 0.4]]))
    train_matrix = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    support = arch_check.vote_support(frames, validation_matrix, train_matrix, k=3)
    assert support["service"][0] == pytest.approx(1.0)
    assert support["service"][1] == pytest.approx(0.6, abs=1e-9)


# --- метрики --------------------------------------------------------------


def test_label_metrics_perfect_prediction():
    metrics = arch_check.label_metrics(["P1", "P2"], ["P1", "P2"], ["P1", "P2"])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_label_metrics_counts_empty_answer_as_error():
    metrics = arch_check.label_metrics(["P1", "P2"], ["P1", ""], ["P1", "P2"])
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"]["P2"] == 0.0


def test_label_metrics_penalises_class_absent_from_prediction():
    metrics = arch_check.label_metrics(["P1", "P3"], ["P3", "P3"], ["P1", "P2", "P3"])
    assert metrics["recall"]["P1"] == 0.0
    assert metrics["f1_macro"] < 1.0


def test_label_metrics_exposes_p1_recall_only_when_p1_known():
    with_p1 = arch_check.label_metrics(["P1"], ["P1"], ["P1", "P2"])
    without = arch_check.label_metrics(["ims"], ["ims"], ["ims", "bss"])
    assert with_p1["p1_recall"] == 1.0
    assert "p1_recall" not in without


# --- кривая порога --------------------------------------------------------


def test_gate_curve_coverage_falls_with_threshold():
    truth = np.array(["P1", "P2", "P3", "P4"])
    pred = np.array(["P1", "P2", "P3", "P4"])
    confidence = np.array([0.9, 0.6, 0.4, 0.2])
    rows = arch_check.gate_curve(truth, pred, confidence, ["P1", "P2", "P3", "P4"])
    coverages = [row["coverage"] for row in rows]
    assert coverages == sorted(coverages, reverse=True)
    assert rows[0]["coverage"] == 1.0


def test_gate_curve_counts_gated_out_p1_as_missed():
    truth = np.array(["P1", "P1"])
    pred = np.array(["P1", "P1"])
    confidence = np.array([0.9, 0.1])
    rows = arch_check.gate_curve(truth, pred, confidence, ["P1"])
    gated = [row for row in rows if row["threshold"] == 0.5][0]
    assert gated["p1_recall"] == 0.5
    assert gated["p1_total"] == 2


def test_gate_curve_returns_none_when_nothing_passes():
    truth = np.array(["P1"])
    pred = np.array(["P1"])
    rows = arch_check.gate_curve(truth, pred, np.array([0.1]), ["P1"])
    empty = [row for row in rows if row["threshold"] == 0.7][0]
    assert empty["covered"] == 0
    assert empty["accuracy"] is None


# --- правило критичности --------------------------------------------------


def test_severity_table_covers_all_phrases():
    table = arch_check.severity_table()
    expected = sum(
        len(scenario.get("impacts_ru", ())) + (1 if scenario.get("impact_en") else 0)
        for scenario in arch_check.SCENARIOS
    )
    assert len(table) == expected
    assert len(table) == 84
    assert {severity for _, severity in table} <= {"P1", "P2", "P3", "P4"}


def test_severity_table_phrases_are_normalised():
    for phrase, _ in arch_check.severity_table():
        assert phrase == arch_check.normalise(phrase)


def test_rule_severity_finds_phrase_case_insensitively():
    table = [("обработка заказов остановлена", "P1")]
    assert arch_check.rule_severity("ОБРАБОТКА   заказов Остановлена", table) == "P1"


def test_rule_severity_prefers_longest_phrase():
    table = [("рост таймаутов", "P3"), ("рост таймаутов diameter", "P2")]
    assert arch_check.rule_severity("наблюдается рост таймаутов diameter", table) == "P2"


def test_rule_severity_returns_none_without_phrase():
    assert arch_check.rule_severity("что-то сломалось", [("падение биллинга", "P1")]) is None


def test_hybrid_severity_keeps_rule_and_fills_gaps():
    hybrid = arch_check.hybrid_severity(["P1", ""], np.array(["P3", "P2"]))
    assert hybrid == ["P1", "P2"]


def test_severity_rule_block_reports_coverage_and_accuracy():
    validation = pd.DataFrame({"description": ["остановлена обработка заказов", "непонятный текст"],
                               "severity": ["P1", "P3"]})
    table = [("остановлена обработка заказов", "P1")]
    block, predicted = arch_check.severity_rule_block(validation, table, ["P1", "P3"])
    assert block["covered"] == 1
    assert block["coverage"] == 0.5
    assert block["accuracy_on_covered"] == 1.0
    assert predicted == ["P1", ""]


# --- выбор по правилу -----------------------------------------------------


def _vote_block(service_f1: float, cause_f1: float, severity_f1: float = 0.4) -> dict:
    targets = {
        "severity": {"f1_macro": severity_f1},
        "service": {"f1_macro": service_f1},
        "root_cause": {"f1_macro": cause_f1},
    }
    return {"3": {"targets": targets}, "5": {"targets": targets}, "10": {"targets": targets}}


def _classifier_block(service_f1: float, cause_f1: float, severity_f1: float = 0.25) -> dict:
    return {
        "targets": {
            "severity": {"f1_macro": severity_f1},
            "service": {"f1_macro": service_f1},
            "root_cause": {"f1_macro": cause_f1},
        }
    }


def test_selection_prefers_hybrid_on_tie():
    selection = arch_check.selection_block(
        _vote_block(0.30, 0.20),
        _classifier_block(0.30, 0.20),
        {"f1_macro_on_all": 0.99},
        {"f1_macro": 0.995},
    )
    assert selection["chosen"]["severity"] == "hybrid"


def test_selection_prefers_simpler_method_within_indifference_band():
    selection = arch_check.selection_block(
        _vote_block(0.305, 0.20),
        _classifier_block(0.300, 0.20),
        {"f1_macro_on_all": 0.99},
        {"f1_macro": 0.99},
    )
    assert selection["chosen"]["service"] == "classifier"


def test_selection_takes_clear_winner_beyond_band():
    selection = arch_check.selection_block(
        _vote_block(0.40, 0.20),
        _classifier_block(0.20, 0.20),
        {"f1_macro_on_all": 0.99},
        {"f1_macro": 0.99},
    )
    assert selection["chosen"]["service"] == "vote"


def test_selection_reports_cost_of_showing_three_neighbours():
    selection = arch_check.selection_block(
        _vote_block(0.40, 0.20),
        _classifier_block(0.20, 0.20),
        {"f1_macro_on_all": 0.99},
        {"f1_macro": 0.99},
    )
    entry = selection["decision"]["service"]
    assert entry["method"] == "vote"
    assert entry["k"] == arch_check.TOP_K
    assert selection["ceiling"]["cost_of_showing_top3"]["service"] == 0.0
    assert selection["ceiling"]["best_k"] == arch_check.TOP_K


def test_selection_lead_is_measured_against_runner_up():
    selection = arch_check.selection_block(
        _vote_block(0.40, 0.20),
        _classifier_block(0.20, 0.20),
        {"f1_macro_on_all": 0.99},
        {"f1_macro": 0.99},
    )
    assert selection["detail"]["service"]["lead_over_runner_up"] == pytest.approx(0.20, abs=1e-9)


# --- согласие соседей -----------------------------------------------------


def test_agreement_block_splits_by_share():
    validation = pd.DataFrame({"service": ["ims", "bss"], "root_cause": ["software", "change"]})
    predicted = {"service": ["ims", "ims"], "root_cause": ["software", "change"]}
    support = {"service": [0.8, 0.2], "root_cause": [0.9, 0.6]}
    classes = {"service": ["ims", "bss"], "root_cause": ["software", "change"]}
    block = arch_check.agreement_block(validation, predicted, support, classes)
    service = block["targets"]["service"]
    assert service["solid_queries"] == 1
    assert service["split_queries"] == 1
    assert service["solid"]["accuracy"] == 1.0
    assert service["split"]["accuracy"] == 0.0


def test_agreement_block_handles_empty_segment():
    validation = pd.DataFrame({"service": ["ims"], "root_cause": ["software"]})
    predicted = {"service": ["ims"], "root_cause": ["software"]}
    support = {"service": [0.9], "root_cause": [0.9]}
    classes = {"service": ["ims"], "root_cause": ["software"]}
    block = arch_check.agreement_block(validation, predicted, support, classes)
    assert block["targets"]["service"]["split_queries"] == 0
    assert block["targets"]["service"]["split"] is None


# --- ворота ---------------------------------------------------------------


def test_gates_mark_unreached_target():
    selection = {"detail": {"severity": {"f1_macro": {"rule": 1.0}},
                            "service": {"f1_macro": {"vote": 0.30}},
                            "root_cause": {"f1_macro": {"classifier": 0.26}}},
                 "chosen": {"severity": "rule", "service": "vote", "root_cause": "classifier"}}
    gates = arch_check.gates_block(selection, {"p1_recall": 0.5}, {"p95_ms": 2.0})
    assert gates["targets_met"] == {"severity": True, "service": False, "root_cause": False}
    assert gates["p1_recall_met"] is False
    assert gates["latency_met"] is True


def test_gates_fail_latency_above_budget():
    selection = {"detail": {"severity": {"f1_macro": {"rule": 1.0}},
                            "service": {"f1_macro": {"vote": 0.30}},
                            "root_cause": {"f1_macro": {"classifier": 0.26}}},
                 "chosen": {"severity": "rule", "service": "vote", "root_cause": "classifier"}}
    gates = arch_check.gates_block(selection, {"p1_recall": 1.0}, {"p95_ms": 1500.0})
    assert gates["latency_met"] is False


def test_gates_treat_missing_p1_recall_as_unmet():
    selection = {"detail": {"severity": {"f1_macro": {"rule": 1.0}},
                            "service": {"f1_macro": {"vote": 0.30}},
                            "root_cause": {"f1_macro": {"classifier": 0.26}}},
                 "chosen": {"severity": "rule", "service": "vote", "root_cause": "classifier"}}
    gates = arch_check.gates_block(selection, {}, {"p95_ms": 2.0})
    assert gates["p1_recall_met"] is False


# --- инструменты и манифест ----------------------------------------------


def test_tools_table_has_python_and_short_licences():
    rows = arch_check.tools_table()
    python_row = [row for row in rows if row["name"] == "python"][0]
    assert python_row["version"].count(".") == 2
    assert python_row["used_in_src"] is True
    assert python_row["used_in_tests"] is True
    for row in rows:
        assert row["role"]
        assert row["status"] in {
            "используется",
            "только в тестах",
            "установлен, напрямую не импортируется",
            "план: не установлен",
            "метаданные не читаются",
        }
        assert isinstance(row["used_in_src"], bool)
        assert isinstance(row["used_in_tests"], bool)
        if row["license"]:
            assert len(str(row["license"])) <= 80


def test_tools_table_marks_missing_package_as_plan():
    rows = arch_check.tools_table()
    for row in rows:
        if not row["installed"]:
            assert row["status"] == "план: не установлен"
            assert row["version"] is None


def test_manifest_check_detects_other_vocabulary():
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer().fit(["совсем другой текст про другое"])
    block = arch_check.manifest_check(vectorizer)
    assert block["checked"] is True
    assert block["match"] is False


def test_manifest_check_reports_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(arch_check, "MANIFEST_PATH", tmp_path / "нет.json")
    block = arch_check.manifest_check(None)
    assert block["checked"] is False
    assert "note" in block


def test_build_report_refuses_other_vocabulary(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"vocabulary_sha256": "0" * 64}), encoding="utf-8")
    monkeypatch.setattr(arch_check, "MANIFEST_PATH", manifest)
    with pytest.raises(ValueError, match="манифест"):
        arch_check.build_report()


def test_build_report_reads_only_train_and_validation(monkeypatch):
    recorded: dict[str, list[str]] = {}

    def fake_loader(splits):
        recorded["splits"] = list(splits)
        raise RuntimeError("stop")

    monkeypatch.setattr(arch_check, "load_selected_splits", fake_loader)
    with pytest.raises(RuntimeError):
        arch_check.build_report()
    assert recorded["splits"] == ["train", "validation"]


# --- задержка и индекс ----------------------------------------------------


def test_index_size_counts_sparse_arrays():
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(np.array([[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0]]))
    block = arch_check.index_size(matrix)
    assert block["rows"] == 2
    assert block["features"] == 5
    assert block["non_zero"] == 3
    assert block["bytes"] > 0


def test_measure_latency_reports_quantiles():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_texts = ["падение биллинга", "отказ ims регистрации", "рост таймаутов diameter"]
    vectorizer = TfidfVectorizer().fit(train_texts)
    matrix = vectorizer.transform(train_texts)
    frames = {
        "train": pd.DataFrame(
            {
                "description": train_texts,
                "severity": ["P1", "P2", "P3"],
                "service": ["billing", "ims", "platform"],
                "root_cause": ["software", "network", "capacity"],
            }
        )
    }
    model = LogisticRegression().fit(matrix, ["a", "b", "c"])
    validation = pd.DataFrame({"description": train_texts})
    table = [("рост таймаутов diameter", "P2")]
    latency = arch_check.measure_latency(frames, validation, vectorizer, matrix, model, table)
    assert latency["queries"] == 3
    assert latency["median_ms"] <= latency["max_ms"]
    assert latency["p95_ms"] <= latency["max_ms"]
    assert latency["rule_misses"] == 2


# --- опубликованные artefacts --------------------------------------------


def test_published_report_shape():
    payload = check_payload()
    assert payload["queries"] == 1200
    assert payload["train_cards"] == 3600
    assert payload["test_used"] is False
    assert payload["selection_split"] == "validation"
    assert set(payload["variants"]) >= {"vote", "classifiers", "severity_rule", "severity_hybrid",
                                        "vote_agreement"}


def test_published_selection_matches_contract_and_gates():
    payload = check_payload()
    assert payload["contract"]["endpoint"] == "POST /predict"
    assert payload["contract"]["request_fields"] == ["title", "description", "monitoring_source", "region"]
    assert payload["selection"]["chosen"] == {"severity": "hybrid", "service": "vote",
                                              "root_cause": "classifier"}
    assert payload["selection"]["decision"]["service"]["k"] == 3
    assert payload["gates"]["targets_met"] == {"severity": True, "service": False,
                                               "root_cause": False}
    assert payload["gates"]["latency_met"] is True


def test_published_report_reproduces_lab9_classifier():
    payload = check_payload()
    assert payload["reference_checks"]["match"] is True
    assert payload["reference_checks"]["accuracy_now"] == payload["reference_checks"]["lab9_reranker_accuracy"]


def test_published_variants_are_ordered_by_known_facts():
    payload = check_payload()
    rule = payload["variants"]["severity_rule"]
    classifiers = payload["variants"]["classifiers"]["targets"]
    assert rule["coverage"] == 1.0
    assert rule["accuracy_on_covered"] == 1.0
    assert classifiers["severity"]["p1_recall"] == 0.0
    assert "P1" not in classifiers["severity"]["predicted"]
    assert "P4" not in classifiers["severity"]["predicted"]
    vote_three = payload["variants"]["vote"]["per_k"]["3"]["targets"]
    assert vote_three["severity"]["p1_recall"] > 0


def test_published_cause_classifier_matches_lab9_saturation():
    payload = check_payload()
    predicted = payload["variants"]["classifiers"]["targets"]["root_cause"]["predicted"]
    assert predicted.get("external_dep", 0) == 0
    assert predicted.get("network", 0) == 0
    share = predicted["capacity"] / payload["queries"]
    assert round(share, 6) == payload["references"]["lab9"]["saturated_share"]


def test_figure_is_published():
    assert FIGURE.exists()
    assert FIGURE.stat().st_size > 40_000


def test_draws_scheme_to_given_path(tmp_path):
    target = tmp_path / "scheme.png"
    assert arch_check.draw_scheme(target) == 0
    assert target.exists()
    assert target.stat().st_size > 40_000


@pytest.mark.parametrize("path", DOCS, ids=[path.name for path in DOCS])
def test_documents_exist_and_mention_decision(path):
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "0,301" in text or "0,3010" in text or "голосование" in text
    assert "ЛР" in text or "Лабораторная 10" in text


def test_matrix_qa_quotes_published_numbers():
    text = (REPO_ROOT / "docs" / "lab10" / "architecture_matrix.md").read_text(encoding="utf-8")
    payload = check_payload()
    vote_three = payload["variants"]["vote"]["per_k"]["3"]["targets"]["service"]["f1_macro"]
    classifier = payload["variants"]["classifiers"]["targets"]["root_cause"]["f1_macro"]
    assert f"{vote_three:.3f}".replace(".", ",") in text
    assert f"{classifier:.3f}".replace(".", ",") in text


def test_decision_document_states_unmet_targets():
    text = (REPO_ROOT / "docs" / "lab10" / "architecture_decision.md").read_text(encoding="utf-8")
    assert "не достигнуты" in text or "не приближается" in text
    assert "Plan B" in text


# --- границы: пустые входы, отказы и NaN ---------------------------------


def doc_text(name: str) -> str:
    return (REPO_ROOT / "docs" / "lab10" / name).read_text(encoding="utf-8")


def ru(value: float, digits: int = 3) -> str:
    """Число в том виде, в каком оно набрано в документах: запятая вместо точки."""
    return f"{value:.{digits}f}".replace(".", ",")


def share(part: float, total: float, digits: int = 1) -> str:
    return f"{part / total * 100:.{digits}f}".replace(".", ",") + " %"


def _selection_stub() -> dict:
    return {
        "detail": {
            "severity": {"f1_macro": {"rule": 1.0}},
            "service": {"f1_macro": {"vote": 0.30}},
            "root_cause": {"f1_macro": {"classifier": 0.26}},
        },
        "chosen": {"severity": "rule", "service": "vote", "root_cause": "classifier"},
    }


def _latency_setup():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = ["падение биллинга", "отказ ims регистрации", "рост таймаутов diameter"]
    vectorizer = TfidfVectorizer().fit(texts)
    matrix = vectorizer.transform(texts)
    frames = {
        "train": pd.DataFrame(
            {
                "description": texts,
                "severity": ["P1", "P2", "P3"],
                "service": ["billing", "ims", "platform"],
                "root_cause": ["software", "network", "capacity"],
            }
        )
    }
    model = LogisticRegression().fit(matrix, ["a", "b", "c"])
    table = [("рост таймаутов diameter", "P2")]
    return frames, vectorizer, matrix, model, table


def test_weighted_vote_ignores_negative_scores():
    label = arch_check.weighted_vote(np.array([-0.5, 0.4]), np.array(["ims", "bss"]), ["ims", "bss"])
    assert label == "bss"


def test_weighted_vote_full_tie_falls_back_to_alphabet():
    label = arch_check.weighted_vote(np.array([0.5, 0.5]), np.array(["ims", "bss"]), ["ims", "bss"])
    assert label == "bss"


def test_weighted_vote_empty_input_returns_none():
    assert arch_check.weighted_vote(np.array([]), np.array([]), ["ims", "bss"]) is None


def test_weighted_vote_skips_nan_score():
    label = arch_check.weighted_vote(
        np.array([np.nan, 0.5, 0.4]), np.array(["ims", "ims", "bss"]), ["ims", "bss"]
    )
    assert label == "ims"


def test_neighbour_order_k_at_most_zero_returns_no_neighbours():
    assert arch_check.neighbour_order(np.array([0.5, 0.4]), 0).size == 0
    assert arch_check.neighbour_order(np.array([0.5, 0.4]), -2).size == 0


def test_neighbour_order_empty_scores_is_empty():
    assert arch_check.neighbour_order(np.array([]), 3).size == 0


def test_neighbour_order_single_score():
    assert list(arch_check.neighbour_order(np.array([0.25]), 3)) == [0]


def test_label_metrics_rejects_empty_samples():
    with pytest.raises(ValueError, match="empty"):
        arch_check.label_metrics([], [], ["P1", "P2"])


def test_label_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError, match="inconsistent"):
        arch_check.label_metrics(["P1", "P2"], ["P1"], ["P1", "P2"])


def test_label_metrics_ignores_labels_outside_taxonomy():
    metrics = arch_check.label_metrics(["P1", "PX"], ["P1", "PX"], ["P1", "P2"])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 0.5
    assert metrics["support"] == {"P1": 1, "P2": 0}


def test_label_metrics_all_empty_predictions():
    metrics = arch_check.label_metrics(["P1", "P2"], ["", ""], ["P1", "P2"])
    assert metrics["accuracy"] == 0.0
    assert metrics["f1_macro"] == 0.0
    assert metrics["support"]["P1"] == 1
    assert metrics["p1_recall"] == 0.0


def test_label_metrics_p1_recall_is_none_without_p1_in_slice():
    metrics = arch_check.label_metrics(["P2", "P3"], ["P2", "P3"], ["P1", "P2", "P3"])
    assert metrics["support"]["P1"] == 0
    assert metrics["p1_recall"] is None


def test_gate_curve_empty_sample_is_zero_coverage():
    rows = arch_check.gate_curve(
        np.array([], dtype=str), np.array([], dtype=str), np.array([]), ["P1", "P2"]
    )
    assert len(rows) == len(arch_check.GATE_THRESHOLDS)
    assert all(row["covered"] == 0 and row["coverage"] == 0.0 for row in rows)
    assert all(row["accuracy"] is None and row["f1_macro"] is None for row in rows)
    assert all(row["p1_recall"] is None for row in rows)


def test_gate_curve_drops_nan_confidence_even_at_zero_threshold():
    rows = arch_check.gate_curve(
        np.array(["P1", "P2"]), np.array(["P1", "P2"]), np.array([np.nan, 0.8]), ["P1", "P2"]
    )
    assert all(row["coverage"] == 0.5 for row in rows)


def test_gate_curve_without_p1_class_has_no_p1_fields():
    rows = arch_check.gate_curve(np.array(["ims"]), np.array(["ims"]), np.array([0.9]), ["ims", "bss"])
    assert all("p1_recall" not in row and "p1_total" not in row for row in rows)


def test_vote_predictions_on_empty_validation_returns_empty_lists():
    from scipy.sparse import csr_matrix

    frames = {
        "train": pd.DataFrame(
            {
                "description": ["a", "b"],
                "severity": ["P1", "P2"],
                "service": ["ims", "bss"],
                "root_cause": ["software", "change"],
            }
        )
    }
    train_matrix = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    predicted, classes, abstained = arch_check.vote_predictions(
        frames, csr_matrix((0, 2)), train_matrix, 3
    )
    assert predicted == {"severity": [], "service": [], "root_cause": []}
    assert abstained == 0
    assert classes == arch_check.TAXONOMY_CLASSES


def test_severity_table_has_no_duplicate_pairs():
    table = arch_check.severity_table()
    assert len(table) == len(set(table))


def test_severity_table_phrase_counts_match_taxonomy():
    """zip в severity_table молча обрезал бы фразы: число последствий и оценок обязано совпасть."""
    for scenario in arch_check.SCENARIOS:
        severities = arch_check.IMPACT_SEVERITY[scenario["id"]]
        assert len(scenario.get("impacts_ru", ())) == len(severities["ru"]), scenario["id"]
        assert bool(scenario.get("impact_en")) == bool(severities.get("en")), scenario["id"]


def test_severity_table_rejects_phrase_count_mismatch(monkeypatch):
    monkeypatch.setattr(arch_check, "SCENARIOS", ({"id": "SC-X", "impacts_ru": ("первая", "вторая")},))
    monkeypatch.setattr(arch_check, "IMPACT_SEVERITY", {"SC-X": {"ru": ("P1",)}})
    with pytest.raises(arch_check.ArchCheckError):
        arch_check.severity_table()


def test_severity_table_rejects_unknown_severity(monkeypatch):
    monkeypatch.setattr(arch_check, "SCENARIOS", ({"id": "SC-X", "impacts_ru": ("фраза",)},))
    monkeypatch.setattr(arch_check, "IMPACT_SEVERITY", {"SC-X": {"ru": ("P9",)}})
    with pytest.raises(arch_check.ArchCheckError):
        arch_check.severity_table()


def test_severity_table_error_is_value_error():
    assert issubclass(arch_check.ArchCheckError, ValueError)


def test_rule_severity_handles_repeated_phrase():
    table = [("рост таймаутов", "P3")]
    assert arch_check.rule_severity("рост таймаутов: снова рост таймаутов", table) == "P3"


def test_rule_severity_needs_matching_punctuation():
    # normalise сохраняет пунктуацию, поэтому запятая внутри фразы ломает совпадение
    table = [("обработка заказов остановлена", "P1")]
    assert arch_check.rule_severity("обработка заказов, остановлена", table) is None


def test_rule_severity_blank_text_returns_none():
    table = [("рост таймаутов", "P3")]
    assert arch_check.rule_severity("", table) is None
    assert arch_check.rule_severity("  \n\t ", table) is None


def test_rule_severity_empty_table_returns_none():
    assert arch_check.rule_severity("рост таймаутов", []) is None


def test_severity_rule_block_zero_coverage_has_no_accuracy():
    validation = pd.DataFrame({"description": ["неизвестный текст"], "severity": ["P2"]})
    block, predicted = arch_check.severity_rule_block(validation, [("рост таймаутов", "P3")], ["P2", "P3"])
    assert block["covered"] == 0
    assert block["coverage"] == 0.0
    assert block["accuracy_on_covered"] is None
    assert block["f1_macro_on_all"] == 0.0
    assert predicted == [""]


def test_severity_rule_block_requires_text_column():
    with pytest.raises(KeyError):
        arch_check.severity_rule_block(pd.DataFrame({"severity": []}), [], ["P1"])


def test_severity_rule_block_empty_frame_raises():
    empty = pd.DataFrame(
        {"description": pd.Series([], dtype=str), "severity": pd.Series([], dtype=str)}
    )
    with pytest.raises(ValueError, match="empty"):
        arch_check.severity_rule_block(empty, [("рост таймаутов", "P3")], ["P1", "P3"])


def test_hybrid_severity_empty_input_returns_empty():
    assert arch_check.hybrid_severity([], np.array([], dtype=str)) == []


def test_hybrid_severity_rejects_length_mismatch():
    with pytest.raises(ValueError):
        arch_check.hybrid_severity(["P1", "P2"], np.array(["P3"]))


def test_selection_requires_the_three_neighbour_block():
    with pytest.raises(KeyError):
        arch_check.selection_block({}, {"targets": {}}, {"f1_macro_on_all": 0.5}, {"f1_macro": 0.5})


def test_selection_requires_key_three_even_if_other_k_exist():
    targets = {
        "severity": {"f1_macro": 0.4},
        "service": {"f1_macro": 0.3},
        "root_cause": {"f1_macro": 0.2},
    }
    vote_per_k = {"5": {"targets": targets}, "10": {"targets": targets}}
    with pytest.raises(KeyError):
        arch_check.selection_block(
            vote_per_k, {"targets": targets}, {"f1_macro_on_all": 0.5}, {"f1_macro": 0.5}
        )


def test_selection_boundary_one_hundredth_prefers_simpler():
    selection = arch_check.selection_block(
        _vote_block(0.31, 0.20),
        _classifier_block(0.30, 0.20),
        {"f1_macro_on_all": 0.99},
        {"f1_macro": 0.99},
    )
    assert selection["chosen"]["service"] == "classifier"


def test_selection_handles_negative_scores():
    selection = arch_check.selection_block(
        _vote_block(-0.10, -0.30),
        _classifier_block(-0.20, -0.30),
        {"f1_macro_on_all": -0.05},
        {"f1_macro": -0.50},
    )
    assert selection["chosen"]["service"] == "vote"
    assert selection["detail"]["service"]["lead_over_runner_up"] == pytest.approx(0.10)


def test_selection_prefers_hybrid_over_classifier_on_tie():
    selection = arch_check.selection_block(
        _vote_block(0.90, 0.20, severity_f1=0.40),
        _classifier_block(0.90, 0.20, severity_f1=0.90),
        {"f1_macro_on_all": 0.40},
        {"f1_macro": 0.90},
    )
    assert selection["chosen"]["severity"] == "hybrid"


def test_selection_reports_adopted_configuration_not_best_k():
    """k=10 лучше k=3, но в ответе три соседа: выбор и ворота обязаны считаться по k=3."""
    def targets(severity: float, service: float, cause: float) -> dict:
        return {
            "severity": {"f1_macro": severity},
            "service": {"f1_macro": service},
            "root_cause": {"f1_macro": cause},
        }

    vote_per_k = {
        "3": {"targets": targets(0.30, 0.30, 0.20)},
        "5": {"targets": targets(0.30, 0.35, 0.25)},
        "10": {"targets": targets(0.30, 0.40, 0.30)},
    }
    classifier_block = {"targets": targets(0.20, 0.25, 0.15)}
    selection = arch_check.selection_block(
        vote_per_k, classifier_block, {"f1_macro_on_all": 0.10}, {"f1_macro": 0.10}
    )

    assert selection["chosen"]["service"] == "vote"
    assert selection["detail"]["service"]["f1_macro"]["vote"] == pytest.approx(0.30)
    assert selection["decision"]["service"]["k"] == arch_check.TOP_K
    assert selection["ceiling"]["best_k"] == 10
    assert selection["ceiling"]["cost_of_showing_top3"]["service"] == pytest.approx(0.10)

    gates = arch_check.gates_block(selection, {"p1_recall": 1.0}, {"p95_ms": 1.0})
    assert gates["adopted_f1_macro"]["service"] == pytest.approx(0.30)
    assert gates["adopted_f1_macro"]["service"] != selection["ceiling"]["by_k"]["10"]["targets"]["service"]


def test_agreement_block_threshold_zero_marks_all_solid():
    validation = pd.DataFrame({"service": ["ims", "bss"], "root_cause": ["software", "change"]})
    predicted = {"service": ["ims", "bss"], "root_cause": ["software", "change"]}
    support = {"service": [0.0, 0.0], "root_cause": [0.0, 0.0]}
    classes = {"service": ["bss", "ims"], "root_cause": ["change", "software"]}
    block = arch_check.agreement_block(validation, predicted, support, classes, threshold=0.0)
    entry = block["targets"]["service"]
    assert entry["solid_queries"] == 2
    assert entry["split"] is None


def test_agreement_block_rejects_threshold_outside_unit_range():
    validation = pd.DataFrame({"service": ["ims"], "root_cause": ["software"]})
    predicted = {"service": ["ims"], "root_cause": ["software"]}
    support = {"service": [0.9], "root_cause": [0.9]}
    classes = {"service": ["ims"], "root_cause": ["software"]}
    for threshold in (1.5, -0.1):
        with pytest.raises(arch_check.ArchCheckError, match="отрезка"):
            arch_check.agreement_block(validation, predicted, support, classes, threshold=threshold)


def test_agreement_block_rejects_length_mismatch():
    validation = pd.DataFrame({"service": ["ims", "bss"], "root_cause": ["software", "change"]})
    predicted = {"service": ["ims", "bss"], "root_cause": ["software", "change"]}
    support = {"service": [0.9], "root_cause": [0.9, 0.9]}
    classes = {"service": ["bss", "ims"], "root_cause": ["change", "software"]}
    with pytest.raises(arch_check.ArchCheckError, match="весов согласия"):
        arch_check.agreement_block(validation, predicted, support, classes)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_agreement_block_empty_input_has_no_nan():
    validation = pd.DataFrame(
        {"service": pd.Series([], dtype=str), "root_cause": pd.Series([], dtype=str)}
    )
    predicted = {"service": [], "root_cause": []}
    support = {"service": [], "root_cause": []}
    classes = {"service": ["ims"], "root_cause": ["software"]}
    block = arch_check.agreement_block(validation, predicted, support, classes)
    assert block["targets"]["service"]["solid_share"] == 0.0
    json.dumps(block, allow_nan=False)


def test_gates_p1_recall_boundary():
    met = arch_check.gates_block(_selection_stub(), {"p1_recall": 0.9}, {"p95_ms": 1.0})
    unmet = arch_check.gates_block(_selection_stub(), {"p1_recall": 0.899999}, {"p95_ms": 1.0})
    assert met["p1_recall_met"] is True
    assert unmet["p1_recall_met"] is False


def test_gates_latency_boundary():
    exact = arch_check.gates_block(_selection_stub(), {"p1_recall": 1.0}, {"p95_ms": 1000.0})
    above = arch_check.gates_block(_selection_stub(), {"p1_recall": 1.0}, {"p95_ms": 1000.001})
    assert exact["latency_met"] is True
    assert above["latency_met"] is False


def test_gates_requires_every_target_in_detail():
    with pytest.raises(arch_check.ArchCheckError, match="принятого варианта"):
        arch_check.gates_block(
            {"detail": {"severity": {"f1_macro": {"rule": 1.0}}}, "chosen": {"severity": "rule"}},
            {"p1_recall": 1.0},
            {"p95_ms": 1.0},
        )


def test_gates_requires_chosen_configuration():
    stub = _selection_stub()
    del stub["chosen"]
    with pytest.raises(arch_check.ArchCheckError, match="detail и chosen"):
        arch_check.gates_block(stub, {"p1_recall": 1.0}, {"p95_ms": 1.0})


def test_gates_marks_missing_latency_as_unmet():
    gates = arch_check.gates_block(_selection_stub(), {"p1_recall": 1.0}, {"p95_ms": None})
    assert gates["latency_met"] is False
    assert gates["latency_p95_ms"] is None


def test_manifest_check_rejects_broken_json(monkeypatch, tmp_path):
    broken = tmp_path / "manifest.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(arch_check, "MANIFEST_PATH", broken)
    with pytest.raises(json.JSONDecodeError):
        arch_check.manifest_check(None)


def test_manifest_check_without_hash_key_does_not_match(monkeypatch, tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"other": 1}), encoding="utf-8")
    monkeypatch.setattr(arch_check, "MANIFEST_PATH", manifest)
    block = arch_check.manifest_check(TfidfVectorizer().fit(["текст про инцидент"]))
    assert block["checked"] is True
    assert block["expected"] is None
    assert block["match"] is False


def test_build_report_refuses_missing_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(arch_check, "MANIFEST_PATH", tmp_path / "нет.json")
    with pytest.raises(ValueError, match="манифест"):
        arch_check.build_report()


def test_measure_latency_single_query_reports_equal_quantiles():
    frames, vectorizer, matrix, model, table = _latency_setup()
    latency = arch_check.measure_latency(
        frames, pd.DataFrame({"description": ["падение биллинга"]}), vectorizer, matrix, model, table
    )
    assert latency["queries"] == 1
    assert latency["median_ms"] == latency["p95_ms"] == latency["max_ms"]
    assert latency["rule_misses"] == 1


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_measure_latency_empty_selection():
    frames, vectorizer, matrix, model, table = _latency_setup()
    latency = arch_check.measure_latency(
        frames,
        pd.DataFrame({"description": pd.Series([], dtype=str)}),
        vectorizer,
        matrix,
        model,
        table,
    )
    assert latency["queries"] == 0


def test_index_size_on_empty_matrix():
    from scipy.sparse import csr_matrix

    block = arch_check.index_size(csr_matrix((0, 5)))
    assert block["rows"] == 0
    assert block["features"] == 5
    assert block["non_zero"] == 0
    assert block["megabytes"] == 0.0


class _PackageMetadata:
    def __init__(self, fields=None, classifiers=None):
        self.fields = fields or {}
        self.classifiers = classifiers

    def get(self, key):
        return self.fields.get(key)

    def get_all(self, key):
        return self.classifiers


def test_licence_prefers_spdx_expression():
    info = _PackageMetadata({"License-Expression": "MIT", "License": "long text"})
    assert arch_check._licence(info) == "MIT"


def test_licence_falls_back_to_classifier():
    info = _PackageMetadata(classifiers=["License :: OSI Approved :: BSD License"])
    assert arch_check._licence(info) == "BSD License"


def test_licence_uses_first_line_and_truncates():
    info = _PackageMetadata({"License": "\n\n" + "x" * 120 + "\nвторая строка"})
    assert arch_check._licence(info) == "x" * 80


def test_licence_reports_absent_metadata():
    assert arch_check._licence(_PackageMetadata()) == "не указана в метаданных"


def test_tools_table_marks_missing_distribution(monkeypatch):
    from importlib import metadata as real_metadata
    from types import SimpleNamespace

    def missing(name):
        raise real_metadata.PackageNotFoundError(name)

    stub = SimpleNamespace(
        version=missing,
        metadata=missing,
        PackageNotFoundError=real_metadata.PackageNotFoundError,
    )
    monkeypatch.setattr(arch_check, "metadata", stub)
    monkeypatch.setattr(arch_check, "TOOLS", (("нет-такого-пакета", "нет_такого_модуля", "роль"),))
    assert arch_check.tools_table() == [
        {
            "name": "нет-такого-пакета",
            "role": "роль",
            "version": None,
            "license": None,
            "installed": False,
            "used_in_src": False,
            "used_in_tests": False,
            "status": "план: не установлен",
        }
    ]


@pytest.mark.parametrize("error", [ValueError("bad metadata"), OSError("bad file"), KeyError("missing")])
def test_tools_table_marks_broken_metadata(monkeypatch, error):
    # битый dist-info не роняет отчёт: строка помечается, таблица достраивается
    from importlib import metadata as real_metadata
    from types import SimpleNamespace

    def broken(name):
        raise error

    stub = SimpleNamespace(
        version=lambda name: "1.0",
        metadata=broken,
        PackageNotFoundError=real_metadata.PackageNotFoundError,
    )
    monkeypatch.setattr(arch_check, "metadata", stub)
    monkeypatch.setattr(arch_check, "_used_modules", lambda: {"src": {"модуль"}, "tests": set()})
    monkeypatch.setattr(arch_check, "TOOLS", (("пакет", "модуль", "роль"),))
    assert arch_check.tools_table() == [
        {
            "name": "пакет",
            "role": "роль",
            "version": None,
            "license": None,
            "installed": True,
            "used_in_src": True,
            "used_in_tests": False,
            "status": "метаданные не читаются",
        }
    ]


def test_tools_table_distinguishes_imported_and_installed(monkeypatch):
    from importlib import metadata as real_metadata
    from types import SimpleNamespace

    info = _PackageMetadata({"License-Expression": "MIT"})
    stub = SimpleNamespace(
        version=lambda name: "1.0",
        metadata=lambda name: info,
        PackageNotFoundError=real_metadata.PackageNotFoundError,
    )
    monkeypatch.setattr(arch_check, "metadata", stub)
    monkeypatch.setattr(
        arch_check, "_used_modules", lambda: {"src": {"sklearn"}, "tests": {"joblib"}}
    )
    monkeypatch.setattr(
        arch_check,
        "TOOLS",
        (
            ("scikit-learn", "sklearn", "модели"),
            ("joblib", "joblib", "сохранение"),
            ("numpy", "numpy", "числа"),
        ),
    )
    rows = arch_check.tools_table()
    assert (rows[0]["status"], rows[0]["used_in_src"], rows[0]["used_in_tests"]) == (
        "используется",
        True,
        False,
    )
    assert (rows[1]["status"], rows[1]["used_in_src"], rows[1]["used_in_tests"]) == (
        "только в тестах",
        False,
        True,
    )
    assert (rows[2]["status"], rows[2]["used_in_src"], rows[2]["used_in_tests"]) == (
        "установлен, напрямую не импортируется",
        False,
        False,
    )


def test_used_modules_reads_imports_from_src_and_tests(monkeypatch, tmp_path):
    src = tmp_path / "src"
    tests = tmp_path / "tests"
    src.mkdir()
    tests.mkdir()
    (src / "arch_check.py").write_text("import numpy\nfrom scipy.sparse import csr_matrix\n", encoding="utf-8")
    (tests / "test_x.py").write_text("import pytest\n", encoding="utf-8")
    monkeypatch.setattr(arch_check, "__file__", str(src / "arch_check.py"))
    assert arch_check._used_modules() == {"src": {"numpy", "scipy"}, "tests": {"pytest"}}


def test_run_writes_report_to_given_directory(monkeypatch, tmp_path):
    payload = {
        "version": "x",
        "selection": {"chosen": {"severity": "hybrid"}},
        "latency": {"median_ms": 1.0, "p95_ms": 2.0},
        "reference_checks": {"checked": True, "match": True},
    }
    monkeypatch.setattr(arch_check, "build_report", lambda: payload)
    assert arch_check.run(tmp_path) == 0
    assert json.loads((tmp_path / "check.json").read_text(encoding="utf-8")) == payload


def test_run_returns_error_and_writes_nothing(monkeypatch, tmp_path):
    def fail():
        raise arch_check.ArchCheckError("словарь не совпал")

    monkeypatch.setattr(arch_check, "build_report", fail)
    assert arch_check.run(tmp_path) == 1
    assert not (tmp_path / "check.json").exists()


def test_run_refuses_report_when_reference_check_failed(monkeypatch, tmp_path):
    for position, reference in enumerate(
        ({"checked": True, "match": False}, {"checked": False})
    ):
        payload = {
            "selection": {"chosen": {}},
            "latency": {"median_ms": 1.0, "p95_ms": 2.0},
            "reference_checks": reference,
        }
        monkeypatch.setattr(arch_check, "build_report", lambda payload=payload: payload)
        target = tmp_path / str(position)
        assert arch_check.run(target) == 1
        assert not (target / "check.json").exists()


def test_main_dispatches_figure(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(arch_check, "draw_scheme", lambda path: calls.setdefault("path", path) and 0)
    monkeypatch.setattr(sys, "argv", ["arch_check.py", "figure", "--path", str(tmp_path / "fig.png")])
    assert arch_check.main() == 0
    assert calls["path"] == tmp_path / "fig.png"


def test_main_dispatches_run(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(arch_check, "run", lambda path: calls.setdefault("path", path) and 0)
    monkeypatch.setattr(sys, "argv", ["arch_check.py", "run", "--results-dir", str(tmp_path)])
    assert arch_check.main() == 0
    assert calls["path"] == tmp_path


# --- опубликованный отчёт: числа выводятся из него, а не из констант ------


def test_published_report_is_strict_json():
    json.dumps(check_payload(), allow_nan=False)


def test_published_selection_recomputes_from_detail():
    payload = check_payload()
    detail = payload["selection"]["detail"]
    order = ("hybrid", "classifier", "vote", "rule")

    def pick(scores: dict[str, float]) -> str:
        best = max(scores.values())
        contenders = [name for name, value in scores.items() if best - value <= 0.01]
        return min(contenders, key=order.index)

    for target in arch_check.TARGETS:
        scores = detail[target]["f1_macro"]
        winner = pick(scores)
        assert detail[target]["chosen"] == winner, target
        assert payload["selection"]["chosen"][target] == winner, target
        assert payload["selection"]["decision"][target]["method"] == winner, target
        runner_up = max(value for name, value in scores.items() if name != winner)
        assert detail[target]["lead_over_runner_up"] == pytest.approx(scores[winner] - runner_up, abs=1e-9)


def test_published_ceiling_is_the_measured_cost_of_showing_three():
    payload = check_payload()
    ceiling = payload["selection"]["ceiling"]
    per_k = payload["variants"]["vote"]["per_k"]
    means = {
        key: np.mean([per_k[key]["targets"][target]["f1_macro"] for target in arch_check.TARGETS])
        for key in per_k
    }
    best = max(arch_check.VOTE_KS, key=lambda k: means[str(k)])
    assert ceiling["best_k"] == best
    assert ceiling["by_k"][str(best)]["mean_f1_macro"] == pytest.approx(means[str(best)], abs=1e-6)
    for target in arch_check.TARGETS:
        expected = per_k[str(best)]["targets"][target]["f1_macro"] - per_k["3"]["targets"][target]["f1_macro"]
        assert ceiling["cost_of_showing_top3"][target] == pytest.approx(expected, abs=1e-9)


def test_published_gates_recompute_from_selection():
    payload = check_payload()
    detail = payload["selection"]["detail"]
    chosen = payload["selection"]["chosen"]
    gates = payload["gates"]
    plan = gates["plan_targets"]
    adopted = {target: detail[target]["f1_macro"][chosen[target]] for target in arch_check.TARGETS}
    assert gates["adopted_f1_macro"] == pytest.approx(adopted, abs=1e-9)
    assert gates["targets_met"] == {target: adopted[target] >= plan[target] for target in arch_check.TARGETS}
    assert gates["latency_met"] is (payload["latency"]["p95_ms"] <= plan["p95_latency_ms"])
    assert gates["p1_recall_met"] is (gates["p1_recall"] >= plan["p1_recall"])
    assert gates["p1_recall"] == payload["variants"]["severity_hybrid"]["p1_recall"]


def test_published_gates_use_adopted_not_best_k():
    payload = check_payload()
    best_k = payload["selection"]["ceiling"]["best_k"]
    if payload["selection"]["chosen"]["service"] != "vote":
        pytest.skip("услуга выбрана не голосованием")
    vote_best = payload["variants"]["vote"]["per_k"][str(best_k)]["targets"]["service"]["f1_macro"]
    assert payload["gates"]["adopted_f1_macro"]["service"] < vote_best


def test_published_agreement_counts_add_up():
    payload = check_payload()
    queries = payload["queries"]
    for target, entry in payload["variants"]["vote_agreement"]["targets"].items():
        assert entry["solid_queries"] + entry["split_queries"] == queries, target
        assert entry["solid_share"] == round(entry["solid_queries"] / queries, 6), target


def test_published_gate_curve_is_monotone():
    payload = check_payload()
    for target, rows in payload["variants"]["classifiers"]["gate"].items():
        covered = [row["covered"] for row in rows]
        assert covered == sorted(covered, reverse=True), target
        for row in rows:
            assert row["coverage"] == round(row["covered"] / payload["queries"], 6), target
            if row["covered"] == 0:
                assert row["accuracy"] is None and row["f1_macro"] is None, target


def test_published_index_matches_features_and_cards():
    payload = check_payload()
    index = payload["index"]
    assert index["rows"] == payload["train_cards"]
    assert index["features"] == payload["features"]["vocabulary_size"]
    assert index["megabytes"] == round(index["bytes"] / 1024 / 1024, 2)
    assert index["non_zero"] <= index["rows"] * index["features"]


def test_published_classes_come_from_taxonomy():
    payload = check_payload()
    assert payload["classes"]["taxonomy"] == arch_check.TAXONOMY_CLASSES
    assert all(not missing for missing in payload["classes"]["missing_in_train"].values())
    assert payload["data_version"] == "1.1"


def test_published_reference_checks_use_cause_classifier_accuracy():
    payload = check_payload()
    accuracy = payload["variants"]["classifiers"]["targets"]["root_cause"]["accuracy"]
    assert payload["reference_checks"]["accuracy_now"] == accuracy
    assert payload["reference_checks"]["match"] is True


def test_published_references_match_lab_files():
    payload = check_payload()
    references = payload["references"]
    lab7 = json.loads((REPO_ROOT / "docs/lab7/results/retrieval_metrics.json").read_text(encoding="utf-8"))
    lab7_classification = json.loads(
        (REPO_ROOT / "docs/lab7/results/classification_metrics.json").read_text(encoding="utf-8")
    )
    lab8 = json.loads((REPO_ROOT / "docs/lab8/results/summary.json").read_text(encoding="utf-8"))
    lab9 = json.loads((REPO_ROOT / "docs/lab9/results/summary.json").read_text(encoding="utf-8"))

    assert references["lab7"]["queries"] == lab7["all"]["queries"]
    assert references["lab7"]["ndcg_at_3_v1"] == lab7["all"]["ndcg_at_k"]
    assert references["lab7"]["service_hit_at_3"] == lab7["all"]["service_hit_at_k"]
    assert references["lab7"]["cause_hit_at_3"] == lab7["all"]["cause_hit_at_k"]
    assert references["lab7"]["both_labels_at_3"] == lab7["all"]["both_labels_at_k"]
    for target in arch_check.TARGETS:
        assert references["lab7"]["majority_f1_macro"][target] == lab7_classification[target]["f1_macro"]

    assert references["lab8"]["ndcg_at_3_v2"] == lab8["ndcg"]["v2"]
    assert references["lab8"]["queries"] == lab8["queries"]
    assert references["lab8"]["p1_gap"] == lab8["p1"]["gap"]
    assert references["lab8"]["alpha"] == lab8["thresholds"]["alpha"]

    assert references["lab9"]["delta_ndcg"] == lab9["delta"]["ndcg@3"]
    assert references["lab9"]["candidate_ndcg"] == lab9["candidate"]["ndcg@3"]
    assert references["lab9"]["reranker_accuracy"] == lab9["reranker"]["accuracy"]
    assert references["lab9"]["saturated_share"] == lab9["reranker"]["saturated_share"]
    assert references["lab9"]["ceiling"] == lab9["reproduction"]["ceiling"]


def test_published_shipped_three_neighbours_still_beat_the_alternative():
    """Выбор считался по k=3, но в ответе всё равно три соседа: порядок обязан сохраниться."""
    payload = check_payload()
    chosen = payload["selection"]["chosen"]
    vote_three = payload["variants"]["vote"]["per_k"][str(arch_check.TOP_K)]["targets"]
    classifiers = payload["variants"]["classifiers"]["targets"]
    for target in ("service", "root_cause"):
        winner = chosen[target]
        loser = "classifier" if winner == "vote" else "vote"
        winner_value = vote_three[target]["f1_macro"] if winner == "vote" else classifiers[target]["f1_macro"]
        loser_value = vote_three[target]["f1_macro"] if loser == "vote" else classifiers[target]["f1_macro"]
        assert winner_value >= loser_value, target


def test_published_hybrid_equals_rule_when_rule_covers_everything():
    payload = check_payload()
    rule = payload["variants"]["severity_rule"]
    hybrid = payload["variants"]["severity_hybrid"]
    if rule["coverage"] == 1.0:
        assert hybrid["f1_macro"] == rule["f1_macro_on_all"]
        assert hybrid["accuracy"] == rule["accuracy_on_covered"] == 1.0


# --- документы: числа берутся из check.json -------------------------------


def test_documents_quote_variant_f1_from_json():
    payload = check_payload()
    vote_three = payload["variants"]["vote"]["per_k"]["3"]["targets"]
    vote_ten = payload["variants"]["vote"]["per_k"]["10"]["targets"]
    classifiers = payload["variants"]["classifiers"]["targets"]

    matrix = doc_text("architecture_matrix.md")
    for source in (vote_three, vote_ten, classifiers):
        for target in arch_check.TARGETS:
            assert ru(source[target]["f1_macro"]) in matrix, (target, source[target]["f1_macro"])

    readme = doc_text("README.md")
    assert ru(payload["variants"]["severity_rule"]["f1_macro_on_all"]) in readme
    assert ru(vote_three["service"]["f1_macro"]) in readme
    assert ru(classifiers["root_cause"]["f1_macro"]) in readme

    decision = doc_text("architecture_decision.md")
    for value in (
        payload["variants"]["severity_rule"]["f1_macro_on_all"],
        classifiers["severity"]["f1_macro"],
        vote_three["service"]["f1_macro"],
        classifiers["service"]["f1_macro"],
        classifiers["root_cause"]["f1_macro"],
        vote_three["root_cause"]["f1_macro"],
    ):
        assert ru(value) in decision


def test_documents_quote_phrase_count_and_coverage_from_json():
    payload = check_payload()
    rule = payload["variants"]["severity_rule"]
    for name in ("architecture_matrix.md", "architecture.md"):
        assert f"{rule['phrases']} фраз" in doc_text(name), name
    if rule["coverage"] == 1.0 and rule["covered"] == payload["queries"]:
        assert "покрытие 100 %" in doc_text("architecture_matrix.md")


def test_documents_quote_gate_numbers_from_json():
    payload = check_payload()
    gate = payload["variants"]["classifiers"]["gate"]["root_cause"]
    threshold = [row for row in gate if row["threshold"] == 0.5][0]
    base = payload["variants"]["classifiers"]["targets"]["root_cause"]
    expected = [
        share(threshold["covered"], payload["queries"]),
        ru(threshold["accuracy"]),
        ru(base["accuracy"]),
        ru(base["f1_macro"]),
        ru(threshold["f1_macro"]),
    ]
    for name in ("architecture_matrix.md", "architecture.md"):
        text = doc_text(name)
        for value in expected:
            assert value in text, (name, value)


def test_documents_quote_agreement_table_from_json():
    payload = check_payload()
    text = doc_text("architecture_matrix.md")
    queries = payload["queries"]
    for target, entry in payload["variants"]["vote_agreement"]["targets"].items():
        for segment in ("solid", "split"):
            assert ru(entry[segment]["accuracy"]) in text, (target, segment)
            assert ru(entry[segment]["f1_macro"]) in text, (target, segment)
        assert share(entry["solid_queries"], queries) in text, target
        assert share(entry["split_queries"], queries) in text, target


def test_documents_quote_index_facts_from_json():
    payload = check_payload()
    index = payload["index"]

    def thousands(value: int) -> str:
        return f"{value:,}".replace(",", " ")

    readme = doc_text("README.md")
    assert thousands(index["rows"]) in readme
    assert thousands(index["features"]) in readme
    assert thousands(index["non_zero"]) in readme
    assert ru(index["megabytes"], 2) in readme

    architecture = doc_text("architecture.md")
    assert thousands(index["rows"]) in architecture
    assert thousands(index["features"]) in architecture
    assert ru(index["megabytes"], 2) in architecture


def test_documents_quote_plan_targets_from_json():
    payload = check_payload()
    targets = payload["gates"]["plan_targets"]
    for name in ("README.md", "architecture_decision.md"):
        text = doc_text(name)
        for target in arch_check.TARGETS:
            assert ru(targets[target], 2) in text, (name, target)
    assert ru(targets["p1_recall"], 2) in doc_text("architecture_decision.md")


def test_decision_table_lists_every_published_tool():
    payload = check_payload()
    text = doc_text("architecture_decision.md").lower()
    for row in payload["tools"]:
        assert row["name"].lower() in text, row["name"]
    if any(not row["installed"] for row in payload["tools"]):
        assert "не установлен" in text

def test_agreement_block_rejects_prediction_length_mismatch():
    validation = pd.DataFrame({"service": ["ims", "bss"], "root_cause": ["software", "change"]})
    predicted = {"service": ["ims"], "root_cause": ["software", "change"]}
    support = {"service": [0.9, 0.9], "root_cause": [0.9, 0.9]}
    classes = {"service": ["ims", "bss"], "root_cause": ["software", "change"]}
    with pytest.raises(arch_check.ArchCheckError, match="предсказаний"):
        arch_check.agreement_block(validation, predicted, support, classes)


def test_gates_reject_malformed_detail_entry():
    selection = {
        "detail": {
            "severity": {"f1_macro": {"rule": 1.0}},
            "service": None,
            "root_cause": {"f1_macro": {"classifier": 0.26}},
        },
        "chosen": {"severity": "rule", "service": "vote", "root_cause": "classifier"},
    }
    with pytest.raises(arch_check.ArchCheckError, match="принятого варианта"):
        arch_check.gates_block(selection, {"p1_recall": 1.0}, {"p95_ms": 1.0})
