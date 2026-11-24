"""Baseline лабораторной 7: метрики, поиск и отчёт собираются по правилам.

Проверяется то, что легко испортить незаметно: правило релевантности и
нормировка NDCG, порядок при равных оценках, поведение на неизвестных словах,
детерминированность подборки случаев и отсутствие утечки: индекс строится
только из train, поэтому карточек validation в выдаче быть не может.

Отдельно закрыты границы метрик (k больше выдачи, пустой идеал, равные оценки),
поведение поиска на пустых и очень длинных запросах, разбор таблицы запросов
на сегменты, сборка отчётов на пустых разрезах, F1-macro с классами, которых
нет в одной из выборок, коды возврата CLI и побайтовое совпадение двух прогонов.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import triage_baseline as baseline_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_train() -> pd.DataFrame:
    """Мини-индекс: четыре карточки с разными услугами и причинами."""
    return pd.DataFrame(
        [
            {
                "incident_id": "I1",
                "scenario_id": "SC-A",
                "language": "ru",
                "service": "ims",
                "root_cause": "capacity",
                "severity": "P2",
                "description": "diameter timeouts grow on ims core",
            },
            {
                "incident_id": "I2",
                "scenario_id": "SC-B",
                "language": "ru",
                "service": "ims",
                "root_cause": "hardware",
                "severity": "P3",
                "description": "diameter board fails on ims core",
            },
            {
                "incident_id": "I3",
                "scenario_id": "SC-C",
                "language": "ru",
                "service": "billing",
                "root_cause": "capacity",
                "severity": "P2",
                "description": "billing queue grows fast",
            },
            {
                "incident_id": "I4",
                "scenario_id": "SC-D",
                "language": "en",
                "service": "bss",
                "root_cause": "change",
                "severity": "P4",
                "description": "crm rejects orders",
            },
        ]
    )


def make_baseline(train: pd.DataFrame | None = None) -> baseline_module.RetrievalBaseline:
    """Индекс на мини-наборе с обычным TfidfVectorizer, без параметров ЛР 6."""
    train = make_train() if train is None else train
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(train["description"])
    return baseline_module.RetrievalBaseline(vectorizer, train, matrix)


def make_queries() -> pd.DataFrame:
    """Два запроса validation, разведённые с индексом по идентификаторам."""
    return pd.DataFrame(
        [
            {
                "incident_id": "V1",
                "scenario_id": "SC-X",
                "language": "ru",
                "service": "ims",
                "root_cause": "capacity",
                "severity": "P2",
                "description": "diameter timeouts on ims core",
            },
            {
                "incident_id": "V2",
                "scenario_id": "SC-Y",
                "language": "ru",
                "service": "ims",
                "root_cause": "capacity",
                "severity": "P2",
                "description": "саксофон лаконичный",
            },
        ]
    )


def test_relevance_grade_covers_four_combinations() -> None:
    """Оценка 2 требует обе метки, оценка 1 — любую одну."""
    candidate = pd.Series({"service": "ims", "root_cause": "capacity"})

    assert baseline_module.relevance_grade("ims", "capacity", candidate) == 2
    assert baseline_module.relevance_grade("ims", "hardware", candidate) == 1
    assert baseline_module.relevance_grade("billing", "capacity", candidate) == 1
    assert baseline_module.relevance_grade("billing", "hardware", candidate) == 0


def test_dcg_uses_exponential_gain_and_log_discount() -> None:
    """DCG позиции: (2^оценка - 1) / log2(позиция + 1)."""
    assert baseline_module.dcg([2, 1, 0], 3) == pytest.approx(3 + 1 / np.log2(3))
    assert baseline_module.dcg([2, 1, 0], 1) == pytest.approx(3)
    assert baseline_module.dcg([], 3) == 0


def test_ndcg_is_one_for_ideal_and_zero_without_relevant() -> None:
    """Идеальный порядок даёт 1, отсутствие релевантных — 0."""
    assert baseline_module.ndcg([2, 1], [2, 1], 3) == pytest.approx(1.0)
    assert baseline_module.ndcg([0, 0, 0], [1, 1, 1], 3) == 0.0
    assert baseline_module.ndcg([1], [0], 3) == 0.0


def test_ideal_grades_fills_with_partial_cards() -> None:
    """Идеал не подставляет оценки, которых в индексе нет."""
    assert baseline_module.ideal_grades(0, 3, 3) == [1, 1, 1]
    assert baseline_module.ideal_grades(5, 0, 3) == [2, 2, 2]
    assert baseline_module.ideal_grades(1, 1, 3) == [2, 1]
    assert baseline_module.ideal_grades(1, 0, 3) == [2]
    assert baseline_module.ideal_grades(0, 0, 3) == []


def test_ndcg_counts_only_positions_up_to_k() -> None:
    """Релевантная карточка ниже k не улучшает метрику."""
    assert baseline_module.ndcg([0, 0, 0, 2], [2, 2, 2], 3) == 0.0
    assert baseline_module.ndcg([1, 0, 0], [1, 1, 1], 3) == pytest.approx(
        1 / (1 + 1 / np.log2(3) + 1 / 2)
    )


def test_majority_class_tie_is_broken_alphabetically() -> None:
    """При равенстве частот константой становится меньшая метка по алфавиту."""
    frame = pd.DataFrame({"service": ["bss", "ims", "bss", "ims", "billing"]})

    assert baseline_module.majority_class(frame, "service") == "bss"


def test_majority_class_takes_the_most_frequent() -> None:
    """Обычный случай: самый частый класс, а не алфавитно первый."""
    frame = pd.DataFrame({"root_cause": ["change", "capacity", "capacity"]})

    assert baseline_module.majority_class(frame, "root_cause") == "capacity"


def test_segment_boundaries_follow_the_published_rule() -> None:
    """Границы сегментов 0, 50 % и 80 % включены в нижнюю сторону."""
    cases = {
        (0, 100): "unknown_all",
        (50, 100): "coverage_low",
        (51, 100): "coverage_mid",
        (80, 100): "coverage_mid",
        (81, 100): "coverage_high",
        (100, 100): "coverage_high",
    }
    for (known, total), expected in cases.items():
        result = baseline_module.SearchResult(hits=(), known_tokens=known, total_tokens=total)
        assert result.segment == expected


def test_segment_without_tokens_is_unknown() -> None:
    """Пустой запрос и запрос из цифр попадают в unknown_all."""
    assert (
        baseline_module.SearchResult(hits=(), known_tokens=0, total_tokens=0).segment
        == "unknown_all"
    )


def test_search_returns_empty_for_unknown_words() -> None:
    """Слова вне словаря не дают совпадений: выдача пуста, а не произвольна."""
    result = make_baseline().search("саксофон лаконичный")

    assert result.hits == ()
    assert result.segment == "unknown_all"
    assert result.coverage == 0.0


def test_search_returns_empty_for_blank_query() -> None:
    """Пробелы и знаки препинания не превращаются в запрос."""
    result = make_baseline().search("   !!!   ")

    assert result.hits == ()
    assert result.total_tokens == 0


def test_search_drops_zero_scores_and_sorts_by_score() -> None:
    """Кандидаты без общих слов не попадают в выдачу, порядок — по убыванию сходства."""
    result = make_baseline().search("diameter timeouts ims core")

    ids = [hit.incident_id for hit in result.hits]
    assert ids[0] == "I1"
    assert set(ids) <= {"I1", "I2"}
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k() -> None:
    """Размер выдачи не превышает k."""
    result = make_baseline().search("diameter ims core", k=1)

    assert len(result.hits) == 1


def test_equal_scores_are_ordered_by_incident_id() -> None:
    """При равных оценках порядок стабилен: по возрастанию идентификатора."""
    train = pd.DataFrame(
        [
            {"incident_id": "I9", "service": "ims", "root_cause": "capacity", "severity": "P2",
             "description": "diameter timeouts on ims core", "scenario_id": "SC", "language": "ru"},
            {"incident_id": "I2", "service": "billing", "root_cause": "software", "severity": "P3",
             "description": "diameter timeouts on ims core", "scenario_id": "SC", "language": "ru"},
        ]
    )
    result = make_baseline(train).search("diameter timeouts", k=2)

    assert [hit.incident_id for hit in result.hits] == ["I2", "I9"]
    assert result.hits[0].score == pytest.approx(result.hits[1].score)


def test_content_tokens_ignore_digits() -> None:
    """Идентификаторы CI и номера не считаются словами при оценке покрытия."""
    engine = make_baseline()

    assert engine.content_tokens("Ошибка CI-429182 на host-spb172, 15530 мс") == {
        "ошибка",
        "ci",
        "на",
        "host",
        "мс",
    }


def test_index_never_contains_validation_cards() -> None:
    """Утечки нет: выдача по запросу validation состоит только из карточек train."""
    engine = make_baseline()
    queries = make_queries()
    train_ids = set(engine.train["incident_id"])
    query_ids = set(queries["incident_id"])

    for _, query in queries.iterrows():
        hits = engine.search(query["description"]).hits
        assert {hit.incident_id for hit in hits} <= train_ids
        assert all(hit.incident_id not in query_ids for hit in hits)


def test_relevant_counts_matches_grades_over_the_index() -> None:
    """Счётчики оценок 2 и 1 согласованы с ручным подсчётом по меткам."""
    engine = make_baseline()

    assert engine.relevant_counts("ims", "capacity") == (1, 2)
    assert engine.relevant_counts("ims", "hardware") == (1, 1)
    assert engine.relevant_counts("billing", "change") == (0, 2)


def test_evaluate_retrieval_returns_one_row_per_query() -> None:
    """Таблица запросов полная и метрики лежат в своих границах."""
    engine = make_baseline()
    per_query = baseline_module.evaluate_retrieval(engine, make_queries())

    assert len(per_query) == 2
    assert set(per_query["incident_id"]) == {"V1", "V2"}
    assert per_query["ndcg"].between(0, 1).all()
    assert set(per_query["recall"]) <= {0, 1}
    assert per_query["returned"].max() <= baseline_module.TOP_K


def test_evaluate_retrieval_marks_zero_vector_query_as_empty() -> None:
    """Запрос из неизвестных слов получает пустую выдачу и нулевые метрики."""
    engine = make_baseline()
    per_query = baseline_module.evaluate_retrieval(engine, make_queries()).set_index("incident_id")
    row = per_query.loc["V2"]

    assert row["returned"] == 0
    assert row["ndcg"] == 0
    assert row["recall"] == 0
    assert row["mrr"] == 0
    assert row["segment"] == "unknown_all"


def test_evaluation_is_deterministic() -> None:
    """Повторный расчёт на том же индексе даёт ту же таблицу."""
    engine = make_baseline()
    first = baseline_module.evaluate_retrieval(engine, make_queries())
    second = baseline_module.evaluate_retrieval(engine, make_queries())

    pd.testing.assert_frame_equal(first, second)


def test_aggregate_handles_empty_segments() -> None:
    """Пустой сегмент не делит на ноль и не роняет отчёт."""
    engine = make_baseline()
    per_query = baseline_module.evaluate_retrieval(engine, make_queries())
    summary = baseline_module.aggregate_retrieval(per_query)

    assert summary["all"]["queries"] == 2
    assert summary["by_segment"]["coverage_low"]["queries"] == 0
    assert summary["by_segment"]["coverage_low"]["ndcg_at_k"] == 0.0
    assert summary["coverage"]["min"] == per_query["coverage"].min()


def test_label_structure_detects_shared_pairs() -> None:
    """Общая пара меток делает оценку 2 достижимой, отсутствие — нет."""
    train = make_train()
    shared = make_train().assign(incident_id=["V1", "V2", "V3", "V4"])
    other = pd.DataFrame(
        [{"incident_id": "V1", "service": "ims", "root_cause": "software"}]
    )

    assert baseline_module.label_structure(train, shared)["grade_two_reachable"] is True
    structure = baseline_module.label_structure(train, other)
    assert structure["grade_two_reachable"] is False
    assert structure["shared_service_cause_pairs"] == 0


def test_pick_cases_is_deterministic_and_spreads_over_scenarios() -> None:
    """Подборка не зависит от порядка строк и берёт карточки из разных сценариев."""
    per_query = pd.DataFrame(
        [
            {"incident_id": f"I{index}", "scenario_id": f"SC-{index % 3}", "ndcg": 1.0}
            for index in range(9)
        ]
    )
    shuffled = per_query.sample(frac=1.0, random_state=7)

    picked = baseline_module.pick_cases(per_query, "ndcg", 1.0, 6)

    assert picked == baseline_module.pick_cases(shuffled, "ndcg", 1.0, 6)
    assert len(picked) == 6
    assert len(set(picked)) == 6
    assert len({scenario for scenario in per_query.set_index("incident_id").loc[picked, "scenario_id"]}) == 3


def test_pick_cases_returns_empty_without_matches() -> None:
    """Если подходящих запросов нет, подборка пуста."""
    per_query = pd.DataFrame([{"incident_id": "I1", "scenario_id": "SC", "ndcg": 0.5}])

    assert baseline_module.pick_cases(per_query, "ndcg", 1.0, 10) == []


def test_classification_reports_unseen_label_with_zero_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Класс, который есть только в train, попадает в отчёт с полнотой 0."""
    monkeypatch.setattr(baseline_module, "TARGETS", ("severity",))
    train = pd.DataFrame({"severity": ["P1", "P2", "P2"]})
    validation = pd.DataFrame({"severity": ["P2", "P3"]})

    report = baseline_module.evaluate_classification(train, validation)["severity"]

    assert report["majority_class"] == "P2"
    assert report["recall_per_class"]["P1"] == 0.0
    assert report["recall_per_class"]["P2"] == 1.0
    assert report["support"]["P1"] == 0
    assert report["accuracy"] == pytest.approx(0.5)


def test_format_value_uses_russian_decimal_comma() -> None:
    """Числа в отчёте записываются с запятой."""
    assert baseline_module.format_value(0.629) == "0,629"
    assert baseline_module.format_value(1.0) == "1,000"


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    """Отпечаток файла повторяется и меняется вместе с содержимым."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    first = baseline_module.sha256_file(path)
    again = baseline_module.sha256_file(path)
    path.write_text("a,b\n1,3\n", encoding="utf-8")

    assert first == again
    assert len(first) == 64
    assert first != baseline_module.sha256_file(path)


def _metrics_fixture() -> tuple[dict[str, object], dict[str, object]]:
    """Минимальные агрегаты для проверки сборки markdown-отчёта."""
    block = {
        "queries": 4,
        "ndcg_at_k": 0.5,
        "recall_at_k": 0.5,
        "full_hit_at_k": 0.0,
        "mrr_at_k": 0.5,
        "service_hit_at_k": 0.5,
        "cause_hit_at_k": 0.25,
        "both_labels_at_k": 0.25,
        "empty_results": 0,
    }
    empty = {**block, "queries": 0, "ndcg_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0,
             "service_hit_at_k": 0.0, "cause_hit_at_k": 0.0, "both_labels_at_k": 0.0,
             "full_hit_at_k": 0.0}
    retrieval = {
        "all": block,
        "by_segment": {
            "coverage_low": block,
            "coverage_mid": empty,
            "coverage_high": empty,
            "unknown_all": empty,
        },
        "by_language": {"ru": block},
        "by_scenario": {"SC-1": block, "SC-2": empty},
        "coverage": {"min": 0.2, "median": 0.4, "max": 0.6, "fully_known_queries": 0},
        "structure": {
            "index_service_cause_pairs": 14,
            "query_service_cause_pairs": 7,
            "shared_service_cause_pairs": 0,
            "grade_two_reachable": False,
            "note": "",
        },
    }
    classification = {
        "severity": {
            "majority_class": "P3",
            "train_share": 0.48,
            "accuracy": 0.37,
            "f1_macro": 0.13,
            "recall_per_class": {"P1": 0.0, "P3": 1.0},
            "support": {"P1": 2, "P3": 2},
            "distinct_classes": 2,
        }
    }
    return retrieval, classification


def test_metrics_table_explains_unreachable_full_relevance(tmp_path: Path) -> None:
    """Если общих пар нет, отчёт прямо говорит, что оценка 2 равна нулю по построению."""
    retrieval, classification = _metrics_fixture()
    path = tmp_path / "metrics.md"

    baseline_module.write_metrics_table(path, retrieval, classification)

    text = path.read_text(encoding="utf-8")
    assert "Общих пар «услуга + причина»" in text
    assert "по построению" in text
    assert "| severity | P3 | 0,480 | 0,370 | 0,130 |" in text


def test_metrics_table_marks_reachable_full_relevance(tmp_path: Path) -> None:
    """При общих парах отчёт не пишет предупреждение о недостижимости."""
    retrieval, classification = _metrics_fixture()
    retrieval["structure"] = {
        **retrieval["structure"],
        "shared_service_cause_pairs": 2,
        "grade_two_reachable": True,
    }
    path = tmp_path / "metrics.md"

    baseline_module.write_metrics_table(path, retrieval, classification)

    text = path.read_text(encoding="utf-8")
    assert "по построению" not in text
    assert "оценка 2 достижима" in text


def test_cases_document_lists_both_sections(tmp_path: Path) -> None:
    """В подборке есть обе части и отметка об отсутствии выдачи."""
    cases = {
        "success": [
            {
                "incident_id": "V1",
                "query": {
                    "description": "diameter timeouts",
                    "service": "ims",
                    "root_cause": "capacity",
                    "severity": "P2",
                    "language": "ru",
                    "segment": "coverage_mid",
                },
                "hits": [
                    {"rank": 1, "incident_id": "I1", "score": 0.42, "grade": 1,
                     "service": "ims", "root_cause": "hardware", "severity": "P3"}
                ],
                "ndcg": 1.0,
                "diagnosis": "первая карточка совпала по метке «услуга», вторая метка не совпала",
                "mechanism": 3,
                "mechanism_text": "общие слова причины и последствия весят больше названия услуги",
                "shared_terms": ["diameter", "timeouts"],
            }
        ],
        "failure": [
            {
                "incident_id": "V2",
                "query": {
                    "description": "саксофон",
                    "service": "ims",
                    "root_cause": "capacity",
                    "severity": "P2",
                    "language": "ru",
                    "segment": "unknown_all",
                },
                "hits": [],
                "ndcg": 0.0,
                "diagnosis": "выдача пуста: ни одно содержательное слово запроса не входит в словарь train",
                "mechanism": 2,
                "mechanism_text": "синонимы и перифраз не пересекаются со словарём train",
                "shared_terms": [],
            }
        ],
    }
    path = tmp_path / "cases.md"

    baseline_module.write_cases(path, cases)

    text = path.read_text(encoding="utf-8")
    assert "## Успешные ответы (1)" in text
    assert "## Проблемные ответы (1)" in text
    assert "| 1 | I1 | 0,420 | 1 |" in text
    assert "| — | выдач нет |" in text
    assert "Причина: выдача пуста" in text
    assert "Слова с наибольшим вкладом в сходство: «diameter», «timeouts»" in text
    assert "Слова с наибольшим вкладом в сходство: —" in text
    assert "Механизм отказа: 2 — синонимы" in text
    assert text.endswith("\n")


def test_run_reports_vocabulary_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Расхождение словаря с манифестом ЛР 6 останавливает прогон до записи отчётов."""
    train = make_train()
    frames = {"train": train, "validation": make_queries(), "test": make_queries()}
    paths = write_split_files(tmp_path / "data", frames)
    manifest = tmp_path / "features_manifest.json"
    manifest.write_text(json.dumps({"vocabulary_sha256": "0" * 64}), encoding="utf-8")
    monkeypatch.setattr(baseline_module, "SPLIT_FILES", paths)
    monkeypatch.setattr(baseline_module, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(baseline_module, "load_splits", lambda paths=None: frames)
    results = tmp_path / "results"

    assert baseline_module.run(results) == 1
    assert not (tmp_path / "metrics.md").exists()
    assert not (results / "settings.json").exists()


def test_committed_settings_match_code_and_data(default_splits: dict[str, pd.DataFrame]) -> None:
    """Опубликованные настройки не расходятся с размером индекса и словарём ЛР 6."""
    stored = json.loads(
        (PROJECT_ROOT / "docs" / "lab7" / "results" / "settings.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (PROJECT_ROOT / "docs" / "lab6" / "features_manifest.json").read_text(encoding="utf-8")
    )

    assert stored["top_k"] == baseline_module.TOP_K
    assert stored["index_split"] == "train"
    assert stored["query_split"] == "validation"
    assert stored["index_size"] == len(default_splits["train"])
    assert stored["vectorizer"]["vocabulary_sha256"] == manifest["vocabulary_sha256"]
    assert stored["seed"] is None


def test_committed_metrics_match_the_results_json() -> None:
    """Числа в metrics.md совпадают с машинным файлом метрик."""
    retrieval = json.loads(
        (PROJECT_ROOT / "docs" / "lab7" / "results" / "retrieval_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    text = (PROJECT_ROOT / "docs" / "lab7" / "metrics.md").read_text(encoding="utf-8")
    overall = retrieval["all"]
    expected = (
        f"| Все запросы | {overall['queries']} | {baseline_module.format_value(overall['ndcg_at_k'])} "
        f"| {baseline_module.format_value(overall['recall_at_k'])} "
        f"| {baseline_module.format_value(overall['full_hit_at_k'])} "
        f"| {baseline_module.format_value(overall['mrr_at_k'])} | {overall['empty_results']} |"
    )

    assert expected in text


def make_engine(frames: dict[str, pd.DataFrame]) -> baseline_module.RetrievalBaseline:
    """Индекс так, как его строит `run`: словарь и матрица считаются только по train."""
    vectorizer = baseline_module.fit_vectorizer(frames["train"])
    matrix = baseline_module.transform(vectorizer, frames["train"])
    return baseline_module.RetrievalBaseline(vectorizer, frames["train"], matrix)


def write_split_files(folder: Path, frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Раскладывает выборки в отдельные csv: `build_settings` берёт из них отпечатки."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, frame in frames.items():
        path = folder / f"{split}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        paths[split] = str(path)
    return paths


def write_manifest(path: Path, train: pd.DataFrame) -> Path:
    """Манифест с отпечатком словаря, который получится из этого train."""
    vectorizer = baseline_module.fit_vectorizer(train)
    path.write_text(
        json.dumps({"vocabulary_sha256": baseline_module.vocabulary_hash(vectorizer)}),
        encoding="utf-8",
    )
    return path


def make_tie_train() -> pd.DataFrame:
    """Три карточки с одинаковым текстом: порядок выдачи задаёт только идентификатор.

    Метки подобраны так, что оценки идут 0, 1, 2 в порядке возрастания id:
    первая релевантная карточка стоит второй, а полностью релевантная — третьей.
    """
    labels = [("I1", "billing", "software"), ("I2", "ims", "software"), ("I3", "ims", "capacity")]
    return pd.DataFrame(
        [
            {
                "incident_id": incident_id,
                "scenario_id": f"SC-{index}",
                "language": "ru",
                "service": service,
                "root_cause": cause,
                "severity": "P2",
                "description": "shared words only",
            }
            for index, (incident_id, service, cause) in enumerate(labels, start=1)
        ]
    )


def make_tie_query() -> pd.DataFrame:
    """Запрос с теми же метками, что у самой релевантной карточки индекса."""
    return pd.DataFrame(
        [
            {
                "incident_id": "V1",
                "scenario_id": "SC-X",
                "language": "ru",
                "service": "ims",
                "root_cause": "capacity",
                "severity": "P2",
                "description": "shared words only",
            }
        ]
    )


# --- метрики: границы DCG, NDCG и идеального списка ---------------------------------


def test_dcg_with_k_beyond_grades_and_zero_k() -> None:
    """k больше числа оценок не добавляет позиций, а k = 0 даёт ноль."""
    assert baseline_module.dcg([2, 1], 10) == pytest.approx(baseline_module.dcg([2, 1], 2))
    assert baseline_module.dcg([2, 1], 0) == 0.0
    assert baseline_module.dcg([], 0) == 0.0


def test_ndcg_is_bounded_and_maximal_only_for_the_ideal_order() -> None:
    """Любой порядок даёт NDCG в [0, 1], единица только у идеального."""
    ideal = baseline_module.ideal_grades(1, 2, 3)
    values = {
        order: baseline_module.ndcg(list(order), ideal, 3)
        for order in ((2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 1, 0), (0, 1, 1), (0, 0, 0))
    }

    assert ideal == [2, 1, 1]
    assert values[(2, 1, 1)] == pytest.approx(1.0)
    assert max(values.values()) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in values.values())
    assert values[(1, 2, 1)] < values[(2, 1, 1)]
    assert values[(1, 1, 2)] < values[(1, 2, 1)]


def test_ndcg_ignores_grades_below_k() -> None:
    """Оценки за пределами k не влияют на метрику."""
    assert baseline_module.ndcg([2, 1, 0], [2, 1, 0], 2) == pytest.approx(1.0)
    assert baseline_module.ndcg([2, 1, 2], [2, 1], 2) == pytest.approx(1.0)


def test_ndcg_accepts_float_grades() -> None:
    """Оценки могут прийти во float: значение метрики не меняется."""
    assert baseline_module.dcg([2.0], 1) == pytest.approx(3.0)
    assert baseline_module.ndcg([2.0, 1.0], [2, 1], 3) == pytest.approx(1.0)


def test_ndcg_without_ideal_or_grades_is_zero() -> None:
    """Пустой идеал и пустая выдача дают ноль, а не деление на ноль."""
    assert baseline_module.ndcg([], [], 3) == 0.0
    assert baseline_module.ndcg([2, 2, 2], [], 3) == 0.0


def test_ideal_grades_respects_k_boundaries() -> None:
    """Идеальный список обрезается по k, а при k = 0 пуст."""
    assert baseline_module.ideal_grades(3, 3, 0) == []
    assert baseline_module.ideal_grades(3, 0, 1) == [2]
    assert baseline_module.ideal_grades(2, 2, 3) == [2, 2, 1]


# --- MRR: позиция первой релевантной карточки ---------------------------------------


def test_mrr_uses_the_first_relevant_position() -> None:
    """Релевантная карточка на второй позиции даёт MRR 0,5, а не 1."""
    engine = make_baseline(make_tie_train())
    row = baseline_module.evaluate_retrieval(engine, make_tie_query()).iloc[0]

    assert [row["top1_grade"], row["service_hit"], row["cause_hit"]] == [0, 1, 1]
    assert row["mrr"] == pytest.approx(0.5)
    assert row["recall"] == 1
    assert row["full_hit"] == 1
    assert row["ndcg"] == pytest.approx((1 / np.log2(3) + 3 / 2) / (3 + 1 / np.log2(3)))


def test_mrr_is_one_for_a_relevant_first_hit() -> None:
    """Если первая карточка релевантна, MRR равен единице."""
    engine = make_baseline()
    row = (
        baseline_module.evaluate_retrieval(engine, make_queries())
        .set_index("incident_id")
        .loc["V1"]
    )

    assert row["top1_id"] == "I1"
    assert row["top1_grade"] == 2
    assert row["mrr"] == 1.0


# --- поиск: границы k, пустые запросы, регистр, длинный текст, равные оценки ---------


def test_search_with_k_larger_than_the_index_returns_every_match() -> None:
    """k больше индекса не ломает выдачу: возвращаются все совпадения."""
    result = make_baseline().search("diameter timeouts", k=99)

    assert [hit.incident_id for hit in result.hits] == ["I1", "I2"]
    assert all(hit.score > 0 for hit in result.hits)


def test_search_with_zero_k_returns_nothing() -> None:
    """k = 0 — пустая выдача, а не весь индекс."""
    assert make_baseline().search("diameter timeouts", k=0).hits == ()


def test_search_without_words_returns_no_hits() -> None:
    """Пустая строка, одни цифры и одни знаки препинания не дают ни одного слова."""
    engine = make_baseline()

    for text in ("", "15530 2026 I000421", "!!! ??? ..."):
        result = engine.search(text)
        assert result.hits == ()
        assert result.total_tokens == 0
        assert result.segment == "unknown_all"


def test_search_folds_cyrillic_case() -> None:
    """Верхний регистр кириллицы не мешает найти карточку и считается известным."""
    train = pd.DataFrame(
        [
            {"incident_id": "I1", "scenario_id": "SC-1", "language": "ru", "service": "ims",
             "root_cause": "capacity", "severity": "P2",
             "description": "рост таймаутов на узле доступа"},
            {"incident_id": "I2", "scenario_id": "SC-2", "language": "ru", "service": "billing",
             "root_cause": "software", "severity": "P3",
             "description": "баланс абонента ушёл в минус"},
        ]
    )
    engine = make_baseline(train)

    result = engine.search("РОСТ ТАЙМАУТОВ")

    assert [hit.incident_id for hit in result.hits] == ["I1"]
    assert engine.content_tokens("РОСТ ТАЙМАУТОВ") == {"рост", "таймаутов"}
    assert result.known_tokens == 2
    assert result.segment == "coverage_high"


def test_search_with_a_long_repeated_query_matches_the_short_one() -> None:
    """Длинный повтор того же текста не меняет ни выдачу, ни покрытие."""
    engine = make_baseline()
    short = engine.search("diameter timeouts")
    long = engine.search("diameter timeouts " * 200)

    assert [hit.incident_id for hit in long.hits] == [hit.incident_id for hit in short.hits]
    assert long.total_tokens == short.total_tokens == 2


def test_tie_order_does_not_depend_on_index_row_order() -> None:
    """При равных оценках порядок задаёт идентификатор, а не порядок строк в индексе."""
    train = make_tie_train()
    forward = make_baseline(train).search("shared words only", k=3)
    backward = make_baseline(train.iloc[::-1].reset_index(drop=True)).search(
        "shared words only", k=3
    )

    assert [hit.incident_id for hit in forward.hits] == ["I1", "I2", "I3"]
    assert [hit.incident_id for hit in backward.hits] == ["I1", "I2", "I3"]
    assert [hit.score for hit in forward.hits] == pytest.approx(
        [hit.score for hit in backward.hits]
    )


# --- утечка: индекс и словарь строятся только по train ------------------------------


def test_search_does_not_extend_the_vocabulary() -> None:
    """Запрос не дообучает словарь: поиск только применяет готовые веса."""
    engine = make_baseline()
    before = dict(engine.vectorizer.vocabulary_)

    engine.search("совершенно неизвестные слова запроса")

    assert engine.vectorizer.vocabulary_ == before


def test_index_matrix_has_one_row_per_train_card() -> None:
    """Матрица индекса совпадает по строкам с train, порядок карточек сохранён."""
    frames = {"train": make_train(), "validation": make_queries()}
    engine = make_engine(frames)

    assert engine.matrix.shape == (len(frames["train"]), len(engine.vectorizer.vocabulary_))
    assert list(engine.train["incident_id"]) == list(frames["train"]["incident_id"])


def test_real_index_returns_only_train_cards(default_splits: dict[str, pd.DataFrame]) -> None:
    """На настоящем наборе выдача по запросам validation состоит только из карточек train."""
    train = default_splits["train"]
    vectorizer = baseline_module.fit_vectorizer(train)
    engine = baseline_module.RetrievalBaseline(
        vectorizer, train, baseline_module.transform(vectorizer, train)
    )
    train_ids = set(train["incident_id"])
    forbidden = set(default_splits["validation"]["incident_id"]) | set(
        default_splits["test"]["incident_id"]
    )

    assert engine.matrix.shape[0] == len(train)
    for text in default_splits["validation"]["description"].head(20):
        hits = {hit.incident_id for hit in engine.search(text).hits}
        assert hits
        assert hits <= train_ids
        assert not hits & forbidden


def test_committed_per_query_rows_are_validation_and_hits_are_train(
    default_splits: dict[str, pd.DataFrame],
) -> None:
    """Опубликованная таблица запросов — это validation, а top-1 в ней — карточка train."""
    per_query = pd.read_csv(
        PROJECT_ROOT / "docs" / "lab7" / "results" / "retrieval_per_query.csv",
        dtype=str,
        keep_default_na=False,
    )
    train_ids = set(default_splits["train"]["incident_id"])
    forbidden = set(default_splits["validation"]["incident_id"]) | set(
        default_splits["test"]["incident_id"]
    )

    assert set(per_query["incident_id"]) == set(default_splits["validation"]["incident_id"])
    assert set(per_query["top1_id"]) - {""} <= train_ids
    assert not (set(per_query["top1_id"]) - {""}) & forbidden


# --- агрегаты: разрезы покрывают все запросы ----------------------------------------


def test_segments_and_languages_partition_all_queries() -> None:
    """Сумма по сегментам и по языкам равна общему числу запросов."""
    engine = make_baseline()
    per_query = baseline_module.evaluate_retrieval(engine, make_queries())
    summary = baseline_module.aggregate_retrieval(per_query)
    by_segment = sum(block["queries"] for block in summary["by_segment"].values())
    by_language = sum(block["queries"] for block in summary["by_language"].values())

    assert by_segment == summary["all"]["queries"]
    assert by_language == summary["all"]["queries"]
    assert summary["by_segment"]["unknown_all"]["queries"] == 1
    assert summary["by_segment"]["coverage_high"]["queries"] == 1
    assert summary["all"]["empty_results"] == 1
    assert summary["coverage"]["fully_known_queries"] == 1
    assert summary["coverage"]["max"] == 1.0


# --- подборка случаев: успех собирается вместе с выдачей -----------------------------


def test_build_cases_collects_a_full_success_with_hits() -> None:
    """Полностью релевантная выдача попадает в успехи вместе с рангами и оценками."""
    train = pd.DataFrame(
        [
            {"incident_id": "I1", "scenario_id": "SC-1", "language": "ru", "service": "ims",
             "root_cause": "capacity", "severity": "P2", "description": "alpha beta"},
            {"incident_id": "I2", "scenario_id": "SC-2", "language": "ru", "service": "ims",
             "root_cause": "capacity", "severity": "P3", "description": "alpha gamma"},
            {"incident_id": "I3", "scenario_id": "SC-3", "language": "ru", "service": "ims",
             "root_cause": "capacity", "severity": "P4", "description": "beta gamma"},
        ]
    )
    queries = pd.DataFrame(
        [{"incident_id": "V1", "scenario_id": "SC-X", "language": "ru", "service": "ims",
          "root_cause": "capacity", "severity": "P2", "description": "alpha beta gamma"}]
    )
    engine = make_baseline(train)
    per_query = baseline_module.evaluate_retrieval(engine, queries)

    cases = baseline_module.build_cases(engine, queries, per_query, limit=5)

    assert cases["failure"] == []
    assert len(cases["success"]) == 1
    entry = cases["success"][0]
    assert entry["incident_id"] == "V1"
    assert entry["ndcg"] == 1.0
    assert [hit["rank"] for hit in entry["hits"]] == [1, 2, 3]
    assert {hit["grade"] for hit in entry["hits"]} == {2}
    assert entry["query"]["segment"] == per_query["segment"].iloc[0]


# --- отчёты: языковой разрез, нулевые метрики, пустые подборки, настройки -----------


def test_metrics_table_lists_every_language_and_zero_rows(tmp_path: Path) -> None:
    """Языковой разрез выводит все языки, а пустой сегмент записывается нулями."""
    retrieval, classification = _metrics_fixture()
    russian = retrieval["all"]
    retrieval["by_language"] = {"en": {**russian, "queries": 2, "ndcg_at_k": 0.75}, "ru": russian}
    path = tmp_path / "metrics.md"

    baseline_module.write_metrics_table(path, retrieval, classification)

    text = path.read_text(encoding="utf-8")
    assert "| en | 2 | 0,750 | 0,500 | 0,500 |" in text
    assert "| ru | 4 | 0,500 | 0,500 | 0,500 |" in text
    assert "| Покрытие словаря 50–80 % | 0 | 0,000 | 0,000 | 0,000 | 0,000 | 0 |" in text


def test_metrics_table_without_languages_keeps_the_section(tmp_path: Path) -> None:
    """Пустой разрез по языкам оставляет заголовок, но не роняет сборку."""
    retrieval, classification = _metrics_fixture()
    retrieval["by_language"] = {}
    path = tmp_path / "metrics.md"

    baseline_module.write_metrics_table(path, retrieval, classification)

    text = path.read_text(encoding="utf-8")
    assert "## Разрез по языку запроса" in text
    assert "| ru |" not in text
    assert "| Все запросы | 4 |" in text


def test_cases_document_handles_empty_sections(tmp_path: Path) -> None:
    """Подборка без случаев записывает обе части с нулём."""
    path = tmp_path / "cases.md"

    baseline_module.write_cases(path, {"success": [], "failure": []})

    text = path.read_text(encoding="utf-8")
    assert "## Успешные ответы (0)" in text
    assert "## Проблемные ответы (0)" in text
    assert "выдач нет" not in text


def test_build_settings_records_hashes_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Настройки фиксируют отпечатки данных, словаря и манифеста, а не только числа."""
    train = make_train()
    frames = {"train": train, "validation": make_queries(), "test": make_queries()}
    paths = write_split_files(tmp_path / "data", frames)
    manifest = write_manifest(tmp_path / "features_manifest.json", train)
    monkeypatch.setattr(baseline_module, "SPLIT_FILES", paths)
    monkeypatch.setattr(baseline_module, "MANIFEST_PATH", manifest)
    engine = make_engine(frames)

    settings = baseline_module.build_settings(engine)

    assert settings["index_size"] == len(train)
    assert settings["index_split"] == "train"
    assert settings["query_split"] == "validation"
    assert settings["top_k"] == baseline_module.TOP_K
    assert settings["seed"] is None
    assert settings["data_sha256"] == {
        split: baseline_module.sha256_file(Path(path)) for split, path in paths.items()
    }
    assert settings["vectorizer"]["vocabulary_size"] == len(engine.vectorizer.vocabulary_)
    assert settings["vectorizer"]["vocabulary_sha256"] == baseline_module.vocabulary_hash(
        engine.vectorizer
    )
    assert settings["vectorizer"]["manifest_sha256"] == settings["vectorizer"]["vocabulary_sha256"]
    assert settings["environment"]["python"].startswith("3.")


def test_build_settings_without_manifest_reports_no_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Без манифеста ЛР 6 настройки собираются, а отпечаток манифеста остаётся пустым."""
    frames = {"train": make_train(), "validation": make_queries(), "test": make_queries()}
    paths = write_split_files(tmp_path / "data", frames)
    monkeypatch.setattr(baseline_module, "SPLIT_FILES", paths)
    monkeypatch.setattr(baseline_module, "MANIFEST_PATH", tmp_path / "missing.json")

    settings = baseline_module.build_settings(make_engine(frames))

    assert settings["vectorizer"]["manifest_sha256"] is None
    assert settings["vectorizer"]["vocabulary_sha256"]


# --- классификация: классы только в train или только в validation -------------------


def test_macro_f1_counts_absent_classes_as_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """F1-macro делится на все классы обеих выборок, включая непредсказанные.

    Константа P2 предсказана обеим карточкам: precision 1/2, recall 1, F1 = 2/3.
    Для P1 и P3 предсказаний нет, поэтому их F1 равны нулю, а macro = (2/3) / 3.
    """
    monkeypatch.setattr(baseline_module, "TARGETS", ("severity",))
    train = pd.DataFrame({"severity": ["P1", "P2", "P2"]})
    validation = pd.DataFrame({"severity": ["P2", "P3"]})

    report = baseline_module.evaluate_classification(train, validation)["severity"]

    assert report["majority_class"] == "P2"
    assert report["distinct_classes"] == 3
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["f1_macro"] == pytest.approx(2 / 9, abs=1e-6)
    assert report["recall_per_class"] == {"P1": 0.0, "P2": 1.0, "P3": 0.0}


def test_validation_only_label_gives_zero_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Класс, которого нет в train, попадает в отчёт поддержкой и обнуляет accuracy."""
    monkeypatch.setattr(baseline_module, "TARGETS", ("severity",))
    train = pd.DataFrame({"severity": ["P2", "P2"]})
    validation = pd.DataFrame({"severity": ["P4"]})

    report = baseline_module.evaluate_classification(train, validation)["severity"]

    assert report["majority_class"] == "P2"
    assert report["train_share"] == 1.0
    assert report["accuracy"] == 0.0
    assert report["f1_macro"] == 0.0
    assert report["support"] == {"P2": 0, "P4": 1}
    assert report["recall_per_class"] == {"P2": 0.0, "P4": 0.0}


def test_majority_class_is_stable_under_row_shuffle() -> None:
    """Tie-break по метке не зависит от порядка строк во входных данных."""
    frame = pd.DataFrame({"service": ["ims", "bss", "bss", "ims", "billing"]})

    answers = {
        baseline_module.majority_class(frame.sample(frac=1.0, random_state=seed), "service")
        for seed in range(5)
    }

    assert answers == {"bss"}


# --- прогон целиком, CLI и обработка ошибок ----------------------------------------


def test_two_runs_write_identical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Два прогона на одних данных дают побайтово одинаковые артефакты."""
    frames = {"train": make_train(), "validation": make_queries(), "test": make_queries()}
    paths = write_split_files(tmp_path / "data", frames)
    manifest = write_manifest(tmp_path / "features_manifest.json", frames["train"])
    monkeypatch.setattr(baseline_module, "SPLIT_FILES", paths)
    monkeypatch.setattr(baseline_module, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(baseline_module, "load_splits", lambda paths=None: frames)
    first = tmp_path / "first" / "results"
    second = tmp_path / "second" / "results"

    assert baseline_module.run(first) == 0
    assert baseline_module.run(second) == 0

    names = sorted(path.name for path in first.iterdir())
    assert names == [
        "cases.json",
        "classification_metrics.json",
        "retrieval_metrics.json",
        "retrieval_per_query.csv",
        "settings.json",
    ]
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first.parent / "metrics.md").read_bytes() == (second.parent / "metrics.md").read_bytes()
    assert (first.parent / "cases.md").read_bytes() == (second.parent / "cases.md").read_bytes()

    per_query = pd.read_csv(first / "retrieval_per_query.csv")
    assert len(per_query) == len(frames["validation"])
    settings = json.loads((first / "settings.json").read_text(encoding="utf-8"))
    assert settings["index_size"] == len(frames["train"])
    assert settings["vectorizer"]["manifest_sha256"] == settings["vectorizer"]["vocabulary_sha256"]


def test_run_reports_missing_data_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Отсутствие файлов выборок — это код 1 и сообщение, а не падение и не пустые отчёты."""

    def missing(*args, **kwargs) -> dict[str, pd.DataFrame]:
        raise FileNotFoundError("нет файла data/processed/incidents_train_v1.1.csv")

    monkeypatch.setattr(baseline_module, "load_splits", missing)
    results = tmp_path / "results"

    assert baseline_module.run(results) == 1
    assert "Ошибка" in capsys.readouterr().out
    assert not (results / "settings.json").exists()
    assert not (tmp_path / "metrics.md").exists()


def test_search_function_uses_the_validation_card(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`search --query-id` берёт текст карточки validation и печатает выдачу."""
    frames = {"train": make_train(), "validation": make_queries(), "test": make_queries()}
    monkeypatch.setattr(baseline_module, "load_splits", lambda paths=None: frames)

    assert baseline_module.search(None, "V1", 3) == 0
    output = capsys.readouterr().out
    assert "запрос: diameter timeouts on ims core" in output
    assert "1. I1 |" in output


def test_search_function_reports_unknown_query_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Неизвестный --query-id — ошибка с кодом 1 и понятным сообщением."""
    frames = {"train": make_train(), "validation": make_queries(), "test": make_queries()}
    monkeypatch.setattr(baseline_module, "load_splits", lambda paths=None: frames)

    assert baseline_module.search(None, "V404", 3) == 1
    assert "в validation нет карточки V404" in capsys.readouterr().out


def test_search_function_without_text_reports_an_empty_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Пустой запрос не ищет ничего и честно об этом пишет."""
    frames = {"train": make_train(), "validation": make_queries(), "test": make_queries()}
    monkeypatch.setattr(baseline_module, "load_splits", lambda paths=None: frames)

    assert baseline_module.search(None, None, 3) == 0
    assert "выдача пуста" in capsys.readouterr().out


def test_main_dispatches_run_with_the_results_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run --results-dir` передаёт путь в `run`, а не пишет в каталог по умолчанию."""
    captured: dict[str, Path] = {}

    def fake_run(path: Path) -> int:
        captured["path"] = path
        return 0

    monkeypatch.setattr(baseline_module, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["triage_baseline.py", "run", "--results-dir", str(tmp_path / "r")]
    )

    assert baseline_module.main() == 0
    assert captured["path"] == tmp_path / "r"


def test_main_returns_error_without_a_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без подкоманды печатается справка и возвращается код 1."""
    monkeypatch.setattr(sys, "argv", ["triage_baseline.py"])

    assert baseline_module.main() == 1


def test_main_rejects_non_positive_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI не принимает k < 1: argparse завершает разбор с кодом 2."""
    monkeypatch.setattr(
        sys, "argv", ["triage_baseline.py", "search", "--query", "diameter", "--k", "0"]
    )

    with pytest.raises(SystemExit) as error:
        baseline_module.main()

    assert error.value.code == 2
