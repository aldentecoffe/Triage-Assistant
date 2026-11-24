"""Тесты работы 9: переранжирование пула, условия решения и границы test.

Проверяется то, что легко сломать незаметно: порядок при равных сигналах,
независимость сортировки от меток запроса, правило решения на границах и
запрет читать test до фиксации решения.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import experiment
from triage_baseline import RetrievalBaseline, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
LAB9 = PROJECT_ROOT / "docs" / "lab9"
RESULTS = LAB9 / "results"

requires_data = pytest.mark.skipif(
    not (PROCESSED / "incidents_train_v1.1.csv").exists(),
    reason="нет файлов данных: сначала пройдите пайплайн",
)
requires_artifacts = pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(),
    reason="нет артефактов работы 9: сначала запустите python src/experiment.py run",
)


def toy_train() -> pd.DataFrame:
    """Шесть карточек train: две услуги, три причины, общие слова."""
    return pd.DataFrame(
        [
            {"incident_id": "T1", "event_id": "E1", "scenario_id": "S1", "language": "ru", "severity": "P2", "service": "s", "root_cause": "c1", "description": "alpha beta gamma"},
            {"incident_id": "T2", "event_id": "E2", "scenario_id": "S1", "language": "ru", "severity": "P3", "service": "s", "root_cause": "c1", "description": "alpha beta delta"},
            {"incident_id": "T3", "event_id": "E3", "scenario_id": "S1", "language": "ru", "severity": "P2", "service": "s", "root_cause": "c2", "description": "alpha gamma epsilon"},
            {"incident_id": "T4", "event_id": "E4", "scenario_id": "S2", "language": "ru", "severity": "P2", "service": "t", "root_cause": "c1", "description": "beta epsilon zeta"},
            {"incident_id": "T5", "event_id": "E5", "scenario_id": "S2", "language": "en", "severity": "P4", "service": "t", "root_cause": "c2", "description": "gamma zeta eta"},
            {"incident_id": "T6", "event_id": "E6", "scenario_id": "S2", "language": "en", "severity": "P3", "service": "u", "root_cause": "c3", "description": "delta epsilon theta"},
        ]
    )


def toy_queries() -> pd.DataFrame:
    """Два запроса: один по своей услуге, второй по редкой причине."""
    return pd.DataFrame(
        [
            {"incident_id": "Q1", "event_id": "E7", "scenario_id": "S3", "language": "ru", "severity": "P2", "service": "s", "root_cause": "c1", "description": "alpha beta gamma"},
            {"incident_id": "Q2", "event_id": "E8", "scenario_id": "S3", "language": "en", "severity": "P3", "service": "u", "root_cause": "c3", "description": "delta epsilon theta"},
        ]
    )


def toy_baseline() -> RetrievalBaseline:
    """Поиск на игрушечном корпусе: тот же класс, что в работе 7."""
    train = toy_train()
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), sublinear_tf=True)
    matrix = vectorizer.fit_transform(train["description"])
    return RetrievalBaseline(vectorizer, train, matrix)


def toy_frame() -> pd.DataFrame:
    """Игрушечный прогон: baseline и кандидат на двух запросах."""
    baseline = toy_baseline()
    reranker = experiment.build_reranker(toy_train(), baseline.matrix)
    return experiment.evaluate_frame(baseline, reranker, toy_queries())


def rows_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Таблица запросов с колонками, которые читает сводка."""
    defaults = {
        "ideal_v2": 3,
        "returned": 3,
        "returned_base": 3,
        "returned_cand": 3,
        "recall_base": 1,
        "recall_cand": 1,
        "cause_hit_base": 1,
        "cause_hit_cand": 1,
        "service_hit_base": 1,
        "service_hit_cand": 1,
        "classifier_top1": 1,
        "predicted_cause": "c1",
        "changed": 1,
        "severity": "P1",
        "language": "ru",
        "root_cause": "c1",
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame["delta"] = (frame["ndcg_cand"] - frame["ndcg_base"]).round(6)
    return frame


def balanced_positive_frame() -> pd.DataFrame:
    """Четыре запроса в двух сценариях: прирост 0,12 без P1 и пустых выдач."""
    rows = [
        {"incident_id": f"Q{index}", "scenario_id": "S1" if index < 2 else "S2", "ndcg_base": 0.4, "ndcg_cand": 0.52}
        for index in range(4)
    ]
    return rows_frame(rows)


def test_top_pool_orders_by_score_then_id() -> None:
    """Равные сходства разрешаются по ID, отрицательные не попадают в пул."""
    ids = pd.array(["A", "B", "C", "D"]).to_numpy()
    scores = pd.array([0.5, 0.5, 0.0, 0.9]).to_numpy()
    pool = experiment.top_pool(scores, ids, pool=3)
    assert list(pool) == [3, 0, 1]


def test_top_pool_cut_and_zero_pool() -> None:
    """Размер пула ограничен, нулевой пул даёт пустой результат."""
    ids = pd.array(["A", "B"]).to_numpy()
    scores = pd.array([0.9, 0.8]).to_numpy()
    assert len(experiment.top_pool(scores, ids, pool=1)) == 1
    assert len(experiment.top_pool(scores, ids, pool=0)) == 0


def test_rerank_puts_higher_probability_first() -> None:
    """Кандидат поднимает карточку с большей вероятностью своей причины."""
    ids = pd.array(["A", "B", "C"]).to_numpy()
    scores = pd.array([0.9, 0.8, 0.7]).to_numpy()
    probability = pd.array([0.1, 0.2, 0.7]).to_numpy()
    order = experiment.rerank_indices(pd.array([0, 1, 2]).to_numpy(), scores, ids, probability, k=3)
    assert list(order) == [2, 1, 0]


def test_rerank_tie_keeps_cosine_then_id() -> None:
    """При равной вероятности решает косинус, затем ID."""
    ids = pd.array(["A", "B", "C"]).to_numpy()
    scores = pd.array([0.5, 0.7, 0.6]).to_numpy()
    probability = pd.array([0.5, 0.5, 0.5]).to_numpy()
    order = experiment.rerank_indices(pd.array([0, 1, 2]).to_numpy(), scores, ids, probability, k=3)
    assert list(order) == [1, 2, 0]


def test_rerank_empty_and_zero_k() -> None:
    """Пустой пул и k = 0 не падают и возвращают пусто."""
    ids = pd.array(["A"]).to_numpy()
    assert len(experiment.rerank_indices(pd.array([], dtype=int), pd.array([0.5]), ids, pd.array([0.5]), k=3)) == 0
    assert len(experiment.rerank_indices(pd.array([0]), pd.array([0.5]), ids, pd.array([0.5]), k=0)) == 0


def test_cause_positions_rejects_unknown() -> None:
    """Причина без обученного класса — ошибка, а не молчаливое смешение."""
    with pytest.raises(ValueError):
        experiment.cause_positions(pd.array(["c1", "c9"]).to_numpy(), pd.array(["c1", "c2"]))


def test_baseline_repeats_lab7_search() -> None:
    """Baseline в эксперименте совпадает с поиском работы 7 на том же корпусе."""
    baseline = toy_baseline()
    frame = toy_frame()
    for _, row in frame.iterrows():
        query = toy_queries().set_index("incident_id").loc[row["incident_id"]]
        result = baseline.search(query["description"])
        expected = " > ".join(hit.incident_id for hit in result.hits)
        assert row["ids_base"] == expected


def test_rerank_does_not_use_query_labels() -> None:
    """Перестановка меток запроса не меняет порядок: сортировка смотрит только текст."""
    baseline = toy_baseline()
    reranker = experiment.build_reranker(toy_train(), baseline.matrix)
    plain = experiment.evaluate_frame(baseline, reranker, toy_queries())

    shuffled = toy_queries().copy()
    shuffled["root_cause"] = shuffled["root_cause"].tolist()[::-1]
    shuffled["service"] = shuffled["service"].tolist()[::-1]
    shuffled["severity"] = shuffled["severity"].tolist()[::-1]
    other = experiment.evaluate_frame(baseline, reranker, shuffled)

    assert list(plain["ids_base"]) == list(other["ids_base"])
    assert list(plain["ids_cand"]) == list(other["ids_cand"])
    # Метки запроса в оценке всё-таки участвуют: если и метрики не изменились,
    # подмена была пустой и тест ничего не проверяет.
    assert list(plain["ndcg_base"]) != list(other["ndcg_base"])
    # Контроль осмысленности подмены: ответ классификатора зависит только от текста
    # запроса, поэтому он обязан совпасть — иначе подмена ничего не проверяет.
    assert list(plain["predicted_cause"]) == list(other["predicted_cause"])


def test_evaluate_frame_is_deterministic() -> None:
    """Два прогона на одном корпусе дают одинаковую таблицу."""
    first = toy_frame()
    second = toy_frame()
    pd.testing.assert_frame_equal(first, second)


def test_summarise_known_values() -> None:
    """Средние считаются по запросам и прирост выводится из них."""
    frame = rows_frame(
        [
            {"incident_id": "Q1", "scenario_id": "S1", "ndcg_base": 1.0, "ndcg_cand": 1.0},
            {"incident_id": "Q2", "scenario_id": "S1", "ndcg_base": 0.5, "ndcg_cand": 1.0},
        ]
    )
    summary = experiment.summarise(frame)
    assert summary["baseline"]["ndcg@3"] == 0.75
    assert summary["candidate"]["ndcg@3"] == 1.0
    assert summary["delta"]["ndcg@3"] == 0.25
    assert summary["changed_queries"] == 2


def test_summarise_empty_frame() -> None:
    """Пустой вход даёт нули, а не исключение и не NaN."""
    summary = experiment.summarise(experiment.empty_per_query())
    assert summary["queries"] == 0
    assert summary["baseline"]["ndcg@3"] == 0.0
    assert summary["protective"]["empty_base"] == 0


def test_decision_accepts_gain_at_threshold() -> None:
    """Прирост ровно 0,10 проходит по правилу «не меньше»."""
    frame = rows_frame(
        [
            {"incident_id": f"Q{index}", "scenario_id": "S1" if index < 2 else "S2", "ndcg_base": 0.4, "ndcg_cand": 0.5}
            for index in range(4)
        ]
    )
    checks = experiment.decision_checks(frame, experiment.summarise(frame))
    assert checks["delta"] == pytest.approx(0.1)
    assert checks["checks"]["gain"] is True
    assert experiment.expected_decision(checks["checks"]) == "include"


def test_decision_rejects_gain_below_threshold() -> None:
    """0,099 — уже отказ: зона неоднозначности объявлена заранее."""
    frame = rows_frame(
        [
            {"incident_id": f"Q{index}", "scenario_id": "S1" if index < 2 else "S2", "ndcg_base": 0.4, "ndcg_cand": 0.499}
            for index in range(4)
        ]
    )
    checks = experiment.decision_checks(frame, experiment.summarise(frame))
    assert checks["checks"]["gain"] is False
    assert experiment.expected_decision(checks["checks"]) == "reject"


def test_decision_rejects_protective_drop() -> None:
    """Падение защитного показателя отменяет прирост главной метрики."""
    frame = balanced_positive_frame().assign(
        severity="P1", cause_hit_base=1, cause_hit_cand=0
    )
    checks = experiment.decision_checks(frame, experiment.summarise(frame))
    assert checks["checks"]["gain"] is True
    assert checks["checks"]["p1_cause"] is False
    assert experiment.expected_decision(checks["checks"]) == "reject"


def test_decision_rejects_single_scenario_gain() -> None:
    """Прирост, целиком пришедший из одного сценария, не считается."""
    frame = rows_frame(
        [
            {"incident_id": "Q1", "scenario_id": "S1", "ndcg_base": 0.4, "ndcg_cand": 0.9},
            {"incident_id": "Q2", "scenario_id": "S2", "ndcg_base": 0.4, "ndcg_cand": 0.4},
            {"incident_id": "Q3", "scenario_id": "S2", "ndcg_base": 0.4, "ndcg_cand": 0.4},
        ]
    )
    checks = experiment.decision_checks(frame, experiment.summarise(frame))
    assert checks["checks"]["gain"] is True
    assert checks["checks"]["spread"] is False
    assert checks["dominant_share"] > experiment.SPREAD_LIMIT


def test_decision_without_gain_has_no_spread_claim() -> None:
    """При нулевом приросте условие распределённости не выдаёт себя за выполненное."""
    frame = rows_frame(
        [{"incident_id": "Q1", "scenario_id": "S1", "ndcg_base": 0.4, "ndcg_cand": 0.4}]
    )
    checks = experiment.decision_checks(frame, experiment.summarise(frame))
    assert checks["checks"]["spread"] is False


def test_uncovered_reports_missing_cause() -> None:
    """Непокрытые причины видны в сводке, а не только в тексте отчёта."""
    frame = rows_frame(
        [{"incident_id": "Q1", "scenario_id": "S1", "ndcg_base": 0.4, "ndcg_cand": 0.5}]
    )
    uncovered = experiment.uncovered_queries(frame, ["c1", "network"])
    assert uncovered["missing_causes"] == ["network"]
    assert uncovered["scenarios"] == 1


def test_uncovered_without_train_causes_does_not_invent_network() -> None:
    """Без списка train причины не выдумываются: пустой список честнее константы."""
    frame = rows_frame(
        [{"incident_id": "Q1", "scenario_id": "S1", "ndcg_base": 0.4, "ndcg_cand": 0.5}]
    )
    assert experiment.uncovered_queries(frame)["missing_causes"] == []


def test_render_comparison_without_test() -> None:
    """До контроля таблица прямо говорит, что test не запускался."""
    frame = toy_frame()
    summary = experiment.summarise(frame)
    text = experiment.render_comparison(summary, frame, None)
    assert "Test не запускался" in text
    assert "## Test: контроль после фиксации решения" not in text


def test_render_decision_mentions_test_control() -> None:
    """После контроля решение ссылается на test и не меняет вердикт."""
    frame = balanced_positive_frame()
    summary = experiment.summarise(frame)
    checks = experiment.decision_checks(frame, summary)
    decision = {
        "decision": "include",
        "reason": "условия выполнены",
        "checks": checks["checks"],
        "dominant_scenario": "S1",
        "dominant_share": 0.5,
    }
    test_summary = experiment.summarise(
        rows_frame(
            [
                {"incident_id": "Q1", "scenario_id": "S9", "ndcg_base": 0.5, "ndcg_cand": 0.45},
                {"incident_id": "Q2", "scenario_id": "S9", "ndcg_base": 0.5, "ndcg_cand": 0.45},
            ]
        )
    )
    text = experiment.render_decision(summary, decision, test_summary)
    assert "## Контроль на test" in text
    assert "Решение E01: **включить**" in text
    assert "прирост -0,050" in text


def test_freeze_refuses_contradiction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Решение не может расходиться с заранее заданными условиями."""
    monkeypatch.setattr(experiment, "render_docs", lambda *args, **kwargs: None)
    frame = balanced_positive_frame()
    summary = experiment.summarise(frame)
    summary["decision_checks"] = experiment.decision_checks(frame, summary)
    summary["config"] = experiment.config_payload()
    (tmp_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit):
        experiment.freeze("reject", "хочу отклонить", results_dir=tmp_path)


def test_freeze_writes_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Согласованное решение записывается вместе с отпечатком сводки."""
    monkeypatch.setattr(experiment, "render_docs", lambda *args, **kwargs: None)
    frame = balanced_positive_frame()
    summary = experiment.summarise(frame)
    summary["decision_checks"] = experiment.decision_checks(frame, summary)
    summary["config"] = experiment.config_payload()
    (tmp_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    experiment.freeze("include", "условия выполнены", results_dir=tmp_path, run_date="2026-11-24")
    payload = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert payload["decision"] == "include"
    assert payload["dominant_share"] == 0.5
    assert len(payload["summary_sha256"]) == 64


def test_test_requires_decision(tmp_path: Path) -> None:
    """До фиксации решения команда test отказывается работать."""
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        experiment.confirm_test(results_dir=tmp_path)


def test_run_never_reads_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Прогон читает только train и validation: test остаётся закрытым."""
    requested: list[list[str]] = []
    frames = {"train": toy_train(), "validation": toy_queries()}

    def fake_load(splits: list[str]) -> dict[str, pd.DataFrame]:
        requested.append(list(splits))
        return {split: frames[split] for split in splits}

    monkeypatch.setattr(experiment, "load_selected_splits", fake_load)
    monkeypatch.setattr(experiment, "check_manifest", lambda *args: None)
    monkeypatch.setattr(experiment, "MANIFEST_PATH", tmp_path / "manifest.json")
    experiment.run(results_dir=tmp_path / "results", docs_dir=tmp_path / "docs")
    assert requested == [["train", "validation"]]
    summary = json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["final_split"] == "test"
    assert (tmp_path / "docs" / "comparison.md").exists()


@requires_artifacts
def test_published_baseline_reproduces_lab8() -> None:
    """Опубликованная сводка повторяет baseline работы 8 и фиксирует отказ."""
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline"]["ndcg@3"] == pytest.approx(0.413059, abs=1e-6)
    assert summary["reproduction"]["match"] is True
    decision = json.loads((RESULTS / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "reject"
    assert decision["checks"] == summary["decision_checks"]["checks"]


@requires_artifacts
def test_plan_config_matches_run() -> None:
    """Конфигурация в карточке гипотезы совпадает с машинной копией прогона."""
    plan = (LAB9 / "experiment_plan.md").read_text(encoding="utf-8")
    start = plan.index("```json") + len("```json")
    end = plan.index("```", start)
    from_plan = json.loads(plan[start:end])
    config = json.loads((RESULTS / "config.json").read_text(encoding="utf-8"))
    assert from_plan == config


@requires_data
def test_real_corpus_grades_are_v2() -> None:
    """На реальном корпусе причина даёт оценку 2 даже при другой услуге."""
    train = pd.read_csv(PROCESSED / "incidents_train_v1.1.csv", dtype=str, keep_default_na=False)
    query = train.iloc[0]
    candidate = query.copy()
    candidate["service"] = "external" if query["service"] != "external" else "bms"
    assert experiment.grade_v2(query, candidate) == 2


# --- Границы: пустая выдача, пустая выборка, повторный прогон ---------------


def frame_with_empty_output() -> pd.DataFrame:
    """Два запроса: у первого нет ни одного положительного сходства с индексом."""
    baseline = toy_baseline()
    reranker = experiment.build_reranker(toy_train(), baseline.matrix)
    queries = toy_queries()
    queries.loc[0, "description"] = "omega"
    return experiment.evaluate_frame(baseline, reranker, queries)


def fast_pipeline(monkeypatch: pytest.MonkeyPatch, validation: pd.DataFrame | None = None) -> None:
    """Подменяет чтение выборок игрушечным корпусом: прогон без данных на диске."""
    frames = {"train": toy_train(), "validation": toy_queries() if validation is None else validation}
    monkeypatch.setattr(
        experiment, "load_selected_splits", lambda splits: {split: frames[split] for split in splits}
    )
    monkeypatch.setattr(experiment, "check_manifest", lambda *args, **kwargs: None)


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Запускает CLI модуля отдельным процессом, как это делает читатель README."""
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "experiment.py"), *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=600,
    )


def test_query_without_positive_similarity_returns_nothing() -> None:
    """Запрос без общих слов с индексом даёт пустую выдачу, а не ошибку и не чужой top-3."""
    frame = frame_with_empty_output()
    row = frame.set_index("incident_id").loc["Q1"]
    assert row["returned"] == 0
    assert row["ids_base"] == "" and row["ids_cand"] == ""
    assert row["grades_base"] == "" and row["grades_cand"] == ""
    assert row["ndcg_base"] == 0.0 and row["ndcg_cand"] == 0.0
    assert row["changed"] == 0
    # Релевантные карточки в индексе есть, поэтому запрос остаётся в средних.
    assert row["ideal_v2"] > 0


def test_summarise_excludes_only_queries_without_relevant_cards() -> None:
    """Из средних выпадают только запросы без релевантных карточек, а пустая выдача — нет."""
    frame = rows_frame(
        [
            {"incident_id": "Q1", "returned": 0, "returned_base": 0, "returned_cand": 0, "ideal_v2": 3, "ndcg_base": 0.0, "ndcg_cand": 0.0},
            {"incident_id": "Q2", "returned": 3, "ideal_v2": 0, "ndcg_base": 0.0, "ndcg_cand": 0.0},
        ]
    )
    summary = experiment.summarise(frame)
    assert summary["queries"] == 2
    assert summary["scored_queries"] == 1
    assert summary["excluded"] == 1
    assert summary["protective"]["empty_base"] == 1
    assert experiment.decision_checks(frame, summary)["checks"]["empty"] is False


def test_empty_per_query_has_the_same_columns_as_a_real_run() -> None:
    """Пустая таблица не расходится по составу колонок с настоящим прогоном."""
    assert list(experiment.empty_per_query().columns) == list(toy_frame().columns)


def test_rerank_returns_whole_pool_when_k_exceeds_it() -> None:
    """k больше пула не обрезает выдачу и не падает."""
    ids = pd.array(["A", "B"]).to_numpy()
    scores = pd.array([0.9, 0.8]).to_numpy()
    probability = pd.array([0.1, 0.9]).to_numpy()
    order = experiment.rerank_indices(pd.array([0, 1]).to_numpy(), scores, ids, probability, k=5)
    assert list(order) == [1, 0]


def test_mean_ndcg_matches_evaluate_frame() -> None:
    """Без оценок сигнала средний NDCG совпадает с baseline прогона."""
    baseline = toy_baseline()
    frame = toy_frame()
    assert experiment.mean_ndcg(baseline, toy_queries()) == pytest.approx(
        float(frame["ndcg_base"].mean()), abs=1e-9
    )


def test_run_refuses_second_experiment_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Второй прогон в тот же каталог не переписывает зафиксированный результат."""
    fast_pipeline(monkeypatch)
    results, docs = tmp_path / "results", tmp_path / "docs"
    experiment.run(results_dir=results, docs_dir=docs)
    (results / "decision.json").write_text("{}", encoding="utf-8")
    before = sha256_file(results / "summary.json")
    with pytest.raises(SystemExit):
        experiment.run(results_dir=results, docs_dir=docs)
    assert sha256_file(results / "summary.json") == before


def test_run_force_starts_new_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force снимает прежнее решение и контроль, чтобы его нельзя было принять за новое."""
    fast_pipeline(monkeypatch)
    results, docs = tmp_path / "results", tmp_path / "docs"
    experiment.run(results_dir=results, docs_dir=docs)
    (results / "decision.json").write_text("{}", encoding="utf-8")
    (results / "test_summary.json").write_text("{}", encoding="utf-8")
    assert experiment.run(results_dir=results, docs_dir=docs, force=True) == 0
    assert not (results / "decision.json").exists()
    assert not (results / "test_summary.json").exists()
    assert (results / "summary.json").exists()


def test_run_force_clears_previous_experiment_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """После --force в каталогах не остаётся файлов прежнего эксперимента."""
    fast_pipeline(monkeypatch)
    results, docs = tmp_path / "results", tmp_path / "docs"
    experiment.run(results_dir=results, docs_dir=docs)
    (results / "test_per_query.csv").write_text("incident_id\nQ1\n", encoding="utf-8")
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "decision.md").write_text("Решение E01: **не включать**.\n", encoding="utf-8")

    experiment.run(results_dir=results, docs_dir=docs, force=True)

    assert not (results / "test_per_query.csv").exists()
    assert not (docs / "decision.md").exists()
    # След прошлого контроля не стирается, а переносится рядом.
    assert (results / "test_per_query.previous.csv").exists()


def test_run_with_empty_validation_writes_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Пустая validation не роняет прогон: сводка и документы нулевые, но собраны."""
    fast_pipeline(monkeypatch, validation=experiment.empty_per_query())
    assert experiment.run(results_dir=tmp_path / "results", docs_dir=tmp_path / "docs") == 0
    summary = json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))
    assert summary["queries"] == 0 and summary["scored_queries"] == 0
    assert (tmp_path / "docs" / "comparison.md").exists()


def test_render_docs_survives_query_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Пустая выдача в per_query.csv не ломает сборку документов и не печатается как nan."""
    monkeypatch.setattr(experiment, "input_hashes", lambda splits: {"train": "0" * 64})
    frame = frame_with_empty_output()
    results, docs = tmp_path / "results", tmp_path / "docs"
    results.mkdir()
    summary = experiment.build_summary(frame)
    (results / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame.to_csv(results / "per_query.csv", index=False)

    experiment.render_docs(results, docs)

    text = (docs / "comparison.md").read_text(encoding="utf-8")
    assert "nan (nan)" not in text


def test_expected_decision_without_checks_is_not_include() -> None:
    """Отсутствие условий не может означать «включать»."""
    assert experiment.expected_decision({}) == "reject"


def test_freeze_without_summary_refuses(tmp_path: Path) -> None:
    """Без прогона фиксировать нечего."""
    with pytest.raises(SystemExit):
        experiment.freeze("reject", "нет сводки", results_dir=tmp_path)


def test_freeze_refuses_incomplete_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сводка с неполным набором условий не даёт зафиксировать решение."""
    monkeypatch.setattr(experiment, "render_docs", lambda *args, **kwargs: None)
    (tmp_path / "summary.json").write_text(
        json.dumps({"decision_checks": {"checks": {"gain": False}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        experiment.freeze("reject", "неполная сводка", results_dir=tmp_path)


def test_freeze_refuses_non_boolean_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Строка «false» вместо False не должна читаться как выполненное условие."""
    monkeypatch.setattr(experiment, "render_docs", lambda *args, **kwargs: None)
    checks = {name: False for name in experiment.CONDITION_TITLES}
    checks["gain"] = "false"  # type: ignore[assignment]
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "decision_checks": {
                    "checks": checks,
                    "rules": {},
                    "dominant_scenario": "S1",
                    "dominant_share": 1.0,
                },
                "baseline": {"ndcg@3": 0.4},
                "candidate": {"ndcg@3": 0.4},
                "delta": {"ndcg@3": 0.0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        experiment.freeze("reject", "не тот тип", results_dir=tmp_path)


def test_test_refuses_changed_summary(tmp_path: Path) -> None:
    """Подменённая после решения сводка отменяет контроль: test перестаёт быть контролем."""
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "decision.json").write_text(
        json.dumps({"decision": "reject", "summary_sha256": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        experiment.confirm_test(results_dir=tmp_path)


def test_signal_sentence_follows_the_signs() -> None:
    """Фраза о сигнале строится по знакам, а не по заготовке «выросло/упало»."""
    summary = {
        "baseline": {"cause_hit@3": 0.5, "service_hit@3": 0.3},
        "candidate": {"cause_hit@3": 0.4, "service_hit@3": 0.4},
    }
    text = experiment.signal_sentence(summary)
    assert "попадание причины в top-3 упало" in text
    assert "попадание услуги выросло" in text
    assert "оплачен" not in text
    assert "Ожидаемого сдвига" in text


def test_saturation_sentence_reports_distribution_of_gain() -> None:
    """Насыщение argmax не объявляет весь прирост эффектом одного класса."""
    summary = {
        "scored_queries": 10,
        "delta": {"ndcg@3": 0.06},
        "reranker": {
            "saturated_share": 0.7,
            "saturated_cause": "capacity",
            "saturated_queries": 7,
            "saturated_gain_share": 0.2,
            "by_cause": [
                {"cause": "capacity", "accuracy": 1.0, "contribution": 0.012},
                {"cause": "change", "accuracy": 0.5, "contribution": 0.03},
                {"cause": "external_dep", "accuracy": 0.0, "contribution": 0.018},
            ],
        },
    }
    text = experiment.saturation_sentence(summary)
    assert "насыщен классом" in text
    assert "не распознана ни разу" in text
    assert "распределён по причинам" in text
    assert "не объясняет весь эффект" in text
    assert "change — 0,030" in text


def test_saturation_sentence_with_concentrated_gain() -> None:
    """Если прирост действительно внутри насыщенного класса, так и говорится."""
    summary = {
        "scored_queries": 10,
        "delta": {"ndcg@3": 0.06},
        "reranker": {
            "saturated_share": 0.8,
            "saturated_cause": "capacity",
            "saturated_queries": 8,
            "saturated_gain_share": 0.7,
            "by_cause": [{"cause": "capacity", "accuracy": 1.0, "contribution": 0.042}],
        },
    }
    text = experiment.saturation_sentence(summary)
    assert "в основном переставляет карточки внутри этого класса" in text


def test_saturation_share_is_none_without_gain() -> None:
    """При нулевом приросте доля насыщенного класса не определена, а не ноль."""
    frame = rows_frame(
        [
            {"incident_id": "Q1", "scenario_id": "S1", "ndcg_base": 0.4, "ndcg_cand": 0.4},
            {"incident_id": "Q2", "scenario_id": "S1", "ndcg_base": 0.4, "ndcg_cand": 0.4},
        ]
    )
    summary = experiment.summarise(frame)
    assert summary["reranker"]["saturated_share"] == 1.0
    assert summary["reranker"]["saturated_gain_share"] is None


def test_saturation_sentence_without_gain_does_not_invent_distribution() -> None:
    """Отрицательный прирост не описывается как «распределённый по причинам»."""
    summary = {
        "scored_queries": 4,
        "delta": {"ndcg@3": -0.008},
        "reranker": {
            "saturated_share": 0.8,
            "saturated_cause": "capacity",
            "saturated_queries": 3,
            "saturated_gain_share": None,
            "by_cause": [{"cause": "capacity", "accuracy": 1.0, "contribution": -0.006}],
        },
    }
    text = experiment.saturation_sentence(summary)
    assert "Прироста нет" in text
    assert "распределён" not in text


def test_protective_block_without_p1() -> None:
    """Без P1 защитный блок честно пишет, что проверять было нечего."""
    protective = {
        "p1_queries": 0,
        "p1_cause_hit_base": 0.0,
        "p1_cause_hit_cand": 0.0,
        "p1_ndcg_base": 0.0,
        "p1_ndcg_cand": 0.0,
        "empty_base": 0,
        "empty_cand": 0,
    }
    text = experiment.protective_block({"protective": protective})
    assert "нет запросов P1" in text
    assert "обнулилось" not in text


def test_confirm_test_refuses_incomplete_checks(tmp_path: Path) -> None:
    """Частичный набор условий не даёт считать test контролем."""
    summary = {"decision_checks": {"checks": {"gain": True}}}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    (tmp_path / "decision.json").write_text(
        json.dumps({"decision": "include", "summary_sha256": digest}), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        experiment.confirm_test(results_dir=tmp_path)


def test_cli_forwards_force_and_results_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI передаёт --force и --results-dir в run, а не теряет их в разборе аргументов."""
    captured: dict[str, object] = {}

    def fake_run(results_dir: Path, docs_dir: Path | None, force: bool) -> int:
        captured.update(results_dir=results_dir, docs_dir=docs_dir, force=force)
        return 0

    monkeypatch.setattr(experiment, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["experiment.py", "run", "--force", "--results-dir", "out/results"]
    )
    assert experiment.main() == 0
    assert captured == {"results_dir": Path("out/results"), "docs_dir": None, "force": True}


def test_cli_run_refuses_when_decision_exists(tmp_path: Path) -> None:
    """Повторный run из командной строки останавливается до чтения данных."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "decision.json").write_text("{}", encoding="utf-8")
    step = run_cli("run", "--results-dir", str(results), "--docs-dir", str(tmp_path / "docs"))
    assert step.returncode == 1
    assert "решение уже зафиксировано" in step.stderr
    assert not (results / "summary.json").exists()


@pytest.fixture(scope="module")
def cli_sequence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Полная цепочка команд на настоящих данных: run, freeze, test в временном каталоге."""
    if not (PROCESSED / "incidents_train_v1.1.csv").exists():
        pytest.skip("нет файлов данных: сначала пройдите пайплайн")
    base = tmp_path_factory.mktemp("lab9_cli")
    results, docs = base / "results", base / "docs"
    common = ("--results-dir", str(results), "--docs-dir", str(docs))
    steps = {"run": run_cli("run", *common)}
    # Решение для freeze берётся из правила, а не вписано числом: тест проверяет
    # CLI, а не конкретный вердикт на конкретных данных.
    expected = "reject"
    if steps["run"].returncode == 0:
        summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
        expected = experiment.expected_decision(summary["decision_checks"]["checks"])
    steps["freeze"] = run_cli(
        "freeze",
        "--decision",
        expected,
        "--reason",
        "тестовое решение на временных каталогах",
        "--date",
        "2026-11-24",
        *common,
    )
    steps["test"] = run_cli("test", *common)
    return {"results": results, "docs": docs, "steps": steps, "expected": expected}


def step_result(cli_sequence: dict[str, object], name: str) -> subprocess.CompletedProcess[str]:
    """Результат шага цепочки; ошибка шага видна сразу, а не как пустой файл ниже."""
    result = cli_sequence["steps"][name]  # type: ignore[index]
    assert result.returncode == 0, result.stderr
    return result


@requires_data
def test_cli_run_reproduces_published_baseline(cli_sequence: dict[str, object]) -> None:
    """CLI повторяет baseline работы 8 и пишет машинные файлы рядом с документами."""
    step_result(cli_sequence, "run")
    results, docs = cli_sequence["results"], cli_sequence["docs"]
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline"]["ndcg@3"] == pytest.approx(0.413059, abs=1e-6)
    assert summary["reproduction"]["match"] is True
    assert sorted(summary["hashes"]) == ["train", "validation"]
    for name in ("per_query.csv", "segments.csv", "config.json", "environment.json"):
        assert (results / name).exists(), name
    assert (docs / "comparison.md").exists()


@requires_data
def test_cli_freeze_records_rule_decision(cli_sequence: dict[str, object]) -> None:
    """freeze пишет решение по правилу вместе с отпечатком validation-сводки."""
    step_result(cli_sequence, "freeze")
    results = cli_sequence["results"]
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    decision = json.loads((results / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == cli_sequence["expected"]
    assert decision["decision"] == experiment.expected_decision(decision["checks"])
    assert decision["checks"] == summary["decision_checks"]["checks"]
    assert decision["summary_sha256"] == sha256_file(results / "summary.json")


@requires_data
def test_cli_test_writes_control_after_decision(cli_sequence: dict[str, object]) -> None:
    """test после freeze пишет контроль на test и не меняет зафиксированное решение."""
    step_result(cli_sequence, "test")
    results, docs = cli_sequence["results"], cli_sequence["docs"]
    control = json.loads((results / "test_summary.json").read_text(encoding="utf-8"))
    decision = json.loads((results / "decision.json").read_text(encoding="utf-8"))
    assert control["queries"] == control["scored_queries"] > 0
    assert control["decision"]["decision"] == decision["decision"]
    assert (results / "test_per_query.csv").exists()
    assert "## Test: контроль после фиксации решения" in (docs / "comparison.md").read_text(
        encoding="utf-8"
    )


@requires_data
def test_run_never_touches_test_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Любое чтение test внутри run упало бы: путь test подменён на несуществующий файл."""
    patched = dict(experiment.SPLIT_FILES)
    patched["test"] = str(tmp_path / "нет_такого_test.csv")
    monkeypatch.setattr(experiment, "SPLIT_FILES", patched)

    assert experiment.run(results_dir=tmp_path / "results", docs_dir=tmp_path / "docs") == 0

    summary = json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))
    assert sorted(summary["hashes"]) == ["train", "validation"]
    assert not list((tmp_path / "results").glob("test*"))


@requires_artifacts
def test_published_protective_p1_dropped_to_zero() -> None:
    """Опубликованный прогон: попадание причины у P1 упало с 0,0625 (6 из 96) до нуля."""
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    protective = summary["protective"]
    assert protective["p1_queries"] == 96
    assert protective["p1_cause_hit_base"] == pytest.approx(6 / 96, abs=1e-6)
    assert protective["p1_cause_hit_cand"] == 0.0
    assert summary["decision_checks"]["checks"]["p1_cause"] is False


@requires_artifacts
def test_published_decision_is_bound_to_summary() -> None:
    """decision.json ссылается на текущую сводку: сводку не подменили после фиксации."""
    decision = json.loads((RESULTS / "decision.json").read_text(encoding="utf-8"))
    assert decision["summary_sha256"] == sha256_file(RESULTS / "summary.json")


@requires_artifacts
def test_published_test_summary_carries_the_same_decision() -> None:
    """Контроль на test сделан после того же решения, что лежит в decision.json."""
    decision = json.loads((RESULTS / "decision.json").read_text(encoding="utf-8"))
    control = json.loads((RESULTS / "test_summary.json").read_text(encoding="utf-8"))
    assert control["decision"] == decision

