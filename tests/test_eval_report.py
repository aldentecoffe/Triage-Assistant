"""Отчёты лабораторной 8: шкала релевантности, разрезы и таксономия.

Проверяется то, что легко испортить незаметно: правило шкалы v2.0 (причина
важнее услуги), состав идеала, нормировка NDCG под идеал [2, 2, 2], границы
сегментов по разнице сигналов и по доле служебных слов, знаменатели в
таксономии, согласованность ручного расчёта с кодом и детерминированность
разрезов. Отдельно закрыто поведение на пустой выдаче и на запросе, у
которого в индексе нет ни одной релевантной карточки.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import eval_report
from triage_baseline import RetrievalBaseline, dcg, ideal_grades, ndcg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED / "incidents_train_v1.1.csv"
VALIDATION_PATH = PROCESSED / "incidents_validation_v1.1.csv"

LAB8 = PROJECT_ROOT / "docs" / "lab8"
RESULTS = LAB8 / "results"
GENERATED_DOCS = ("segments.md", "error_taxonomy.md", "manual_calculation.md")

requires_data = pytest.mark.skipif(
    not (TRAIN_PATH.exists() and VALIDATION_PATH.exists()),
    reason="нет файлов данных: сначала пройдите пайплайн",
)

requires_artifacts = pytest.mark.skipif(
    not all((RESULTS / name).exists() for name in ("summary.json", "manual_case.json", "per_query.csv", "errors.csv"))
    or not all((LAB8 / name).exists() for name in GENERATED_DOCS),
    reason="нет артефактов ЛР 8: сначала `python src/eval_report.py run`",
)


def read_doc(name: str) -> str:
    """Текст опубликованного документа ЛР 8."""
    return (LAB8 / name).read_text(encoding="utf-8")


def parse_number(cell: str) -> float:
    """Русская запись числа из markdown обратно в float."""
    return float(cell.strip().replace("\u00a0", "").replace(" ", "").replace(",", "."))


def table_rows(text: str) -> list[list[str]]:
    """Строки markdown-таблиц без заголовков-разделителей."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(cell and set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def manual_payload(grades: list[int], counts: tuple[int, int] = (5, 5)) -> dict[str, object]:
    """Словарь для render_manual в том виде, в каком его отдаёт manual_case."""
    ideal = ideal_grades(*counts, 3)
    return {
        "incident_id": "Q9",
        "text": "проверочный запрос",
        "service": "ims",
        "root_cause": "capacity",
        "severity": "P2",
        "hits": [
            {"rank": rank, "incident_id": f"T{rank}", "score": 0.9 / rank, "grade": grade}
            for rank, grade in enumerate(grades, start=1)
        ],
        "grades": grades,
        "ideal_grades": ideal,
        "dcg": round(dcg(grades, 3), 5),
        "idcg": round(dcg(ideal, 3), 5),
        "ndcg": round(ndcg(grades, ideal, 3), 6),
        "mrr": 1.0,
        "counts": {"grade_two": counts[0], "grade_one": counts[1]},
    }


def small_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Мини-корпус, уже посчитанный evaluate."""
    train, validation = make_frames()
    return eval_report.evaluate(build_baseline(train), validation), validation


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Мини-корпус: три карточки train и два запроса."""
    train = pd.DataFrame(
        [
            {
                "incident_id": "I1",
                "scenario_id": "SC-A",
                "event_id": "E1",
                "language": "ru",
                "service": "ims",
                "root_cause": "capacity",
                "severity": "P2",
                "description": "рост таймаутов Diameter на узле связи",
            },
            {
                "incident_id": "I2",
                "scenario_id": "SC-B",
                "event_id": "E2",
                "language": "ru",
                "service": "ims",
                "root_cause": "software",
                "severity": "P3",
                "description": "ошибка обработки запросов в модуле регистрации",
            },
            {
                "incident_id": "I3",
                "scenario_id": "SC-C",
                "event_id": "E3",
                "language": "ru",
                "service": "billing",
                "root_cause": "capacity",
                "severity": "P4",
                "description": "перегрузка биллинга на пике нагрузки",
            },
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "incident_id": "Q1",
                "scenario_id": "SC-D",
                "event_id": "E4",
                "language": "ru",
                "service": "ims",
                "root_cause": "capacity",
                "severity": "P1",
                "description": "таймауты Diameter выросли на узле",
            },
            {
                "incident_id": "Q2",
                "scenario_id": "SC-E",
                "event_id": "E5",
                "language": "ru",
                "service": "transport",
                "root_cause": "network",
                "severity": "P2",
                "description": "обрыв оптики между площадками",
            },
        ]
    )
    return train, validation


def build_baseline(train: pd.DataFrame) -> RetrievalBaseline:
    """Векторизатор и матрица на мини-корпусе, без файлов ЛР 6."""
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    vectorizer.fit(train["description"])
    matrix = vectorizer.transform(train["description"])
    return RetrievalBaseline(vectorizer, train, matrix)


class TestGrading:
    def test_same_cause_any_service_is_two(self) -> None:
        query = {"service": "ims", "root_cause": "capacity"}
        candidate = {"service": "billing", "root_cause": "capacity"}
        assert eval_report.grade_v2(query, candidate) == 2

    def test_same_service_other_cause_is_one(self) -> None:
        query = {"service": "ims", "root_cause": "capacity"}
        candidate = {"service": "ims", "root_cause": "software"}
        assert eval_report.grade_v2(query, candidate) == 1

    def test_no_label_match_is_zero(self) -> None:
        query = {"service": "ims", "root_cause": "capacity"}
        candidate = {"service": "billing", "root_cause": "hardware"}
        assert eval_report.grade_v2(query, candidate) == 0

    def test_cause_beats_service_when_both_match(self) -> None:
        query = {"service": "ims", "root_cause": "capacity"}
        candidate = {"service": "ims", "root_cause": "capacity"}
        assert eval_report.grade_v2(query, candidate) == 2

    def test_old_scale_needs_both_labels(self) -> None:
        query = {"service": "ims", "root_cause": "capacity"}
        assert eval_report.grade_v1(query, {"service": "billing", "root_cause": "capacity"}) == 1

    def test_new_scale_is_not_looser_than_old_for_zero(self) -> None:
        query = {"service": "ims", "root_cause": "capacity"}
        candidate = {"service": "billing", "root_cause": "hardware"}
        assert eval_report.grade_v1(query, candidate) == eval_report.grade_v2(query, candidate) == 0


class TestIdealAndMetric:
    def test_ideal_counts_split_by_label(self) -> None:
        train, validation = make_frames()
        baseline = build_baseline(train)
        query = validation.iloc[0]
        counts = eval_report.ideal_counts(baseline, query)["v2"]
        assert counts == (2, 1)

    def test_ideal_counts_old_scale_has_no_grade_two(self) -> None:
        train, validation = make_frames()
        baseline = build_baseline(train)
        counts = eval_report.ideal_counts(baseline, validation.iloc[1])["v1"]
        assert counts == (0, 0)

    def test_ideal_is_padded_with_grade_two_when_cause_exists(self) -> None:
        train, validation = make_frames()
        baseline = build_baseline(train)
        from triage_baseline import ideal_grades

        counts = eval_report.ideal_counts(baseline, validation.iloc[0])["v2"]
        ideal = ideal_grades(*counts, 3)
        assert ideal[:2] == [2, 2]
        assert len(ideal) == 3

    def test_perfect_order_gives_one(self) -> None:
        from triage_baseline import ideal_grades, ndcg

        assert ndcg([2, 2, 2], ideal_grades(5, 5, 3), 3) == pytest.approx(1.0)

    def test_same_service_three_times_gives_one_third(self) -> None:
        from triage_baseline import ideal_grades, ndcg

        assert ndcg([1, 1, 1], ideal_grades(5, 5, 3), 3) == pytest.approx(0.3333, abs=1e-4)

    def test_order_matters(self) -> None:
        from triage_baseline import ideal_grades, ndcg

        ideal = ideal_grades(5, 5, 3)
        assert ndcg([2, 1, 0], ideal, 3) > ndcg([1, 2, 0], ideal, 3)

    def test_empty_output_scores_zero(self) -> None:
        from triage_baseline import ideal_grades, ndcg

        assert ndcg([], ideal_grades(5, 5, 3), 3) == 0.0

    def test_zero_k_scores_zero(self) -> None:
        from triage_baseline import ideal_grades, ndcg

        assert ndcg([2, 2, 2], ideal_grades(5, 5, 3), 0) == 0.0


class TestBands:
    @pytest.mark.parametrize(
        ("gap", "expected"),
        [
            (0.5, "cause_closer"),
            (0.02, "cause_closer"),
            (0.01, "signal_tie"),
            (0.0, "signal_tie"),
            (-0.01, "signal_tie"),
            (-0.02, "service_closer"),
            (-0.5, "service_closer"),
        ],
    )
    def test_signal_band_boundaries(self, gap: float, expected: str) -> None:
        assert eval_report.signal_band(gap) == expected

    @pytest.mark.parametrize(
        ("share", "expected"),
        [
            (0.0, "noise_low"),
            (0.30, "noise_low"),
            (0.31, "noise_mid"),
            (0.50, "noise_mid"),
            (0.51, "noise_high"),
        ],
    )
    def test_noise_band_boundaries(self, share: float, expected: str) -> None:
        assert eval_report.noise_band(share) == expected


class TestEvaluate:
    def test_one_row_per_query(self) -> None:
        train, validation = make_frames()
        baseline = build_baseline(train)
        frame = eval_report.evaluate(baseline, validation)
        assert len(frame) == len(validation)
        assert set(frame["incident_id"]) == set(validation["incident_id"])

    def test_ndcg_within_bounds(self) -> None:
        train, validation = make_frames()
        frame = eval_report.evaluate(build_baseline(train), validation)
        assert frame["ndcg_v2"].between(0, 1).all()

    def test_cause_query_finds_cause_card(self) -> None:
        train, validation = make_frames()
        frame = eval_report.evaluate(build_baseline(train), validation)
        row = frame.set_index("incident_id").loc["Q1"]
        assert row["cause_exact_hit"] == 1

    def test_matching_query_returns_cards(self) -> None:
        train, validation = make_frames()
        frame = eval_report.evaluate(build_baseline(train), validation)
        row = frame.set_index("incident_id").loc["Q1"]
        assert row["returned"] >= 1

    def test_uncovered_query_gets_zero_and_no_crash(self) -> None:
        train, validation = make_frames()
        validation = validation.copy()
        validation.loc[0, "description"] = "zzz qqq"
        frame = eval_report.evaluate(build_baseline(train), validation)
        row = frame.set_index("incident_id").loc["Q1"]
        assert row["returned"] <= 1
        assert row["ndcg_v2"] == 0.0

    def test_cascade_flag_uses_validation_events(self) -> None:
        train, validation = make_frames()
        extra = validation.iloc[[0]].copy()
        extra["incident_id"] = "Q3"
        validation = pd.concat([validation, extra], ignore_index=True)
        frame = eval_report.evaluate(build_baseline(train), validation)
        flags = frame.set_index("incident_id")["cascade"]
        assert flags["Q1"] == flags["Q3"] == "multi"
        assert flags["Q2"] == "single"

    def test_evaluate_is_deterministic(self) -> None:
        train, validation = make_frames()
        baseline = build_baseline(train)
        first = eval_report.evaluate(baseline, validation)
        second = eval_report.evaluate(baseline, validation)
        pd.testing.assert_frame_equal(first, second)


class TestAggregate:
    def frame(self) -> pd.DataFrame:
        train, validation = make_frames()
        return eval_report.evaluate(build_baseline(train), validation)

    def test_overall_row_counts_scored_queries(self) -> None:
        table = eval_report.aggregate(self.frame())
        assert table.loc[0, "queries"] == 1
        assert table.loc[0, "label"] == "Все запросы"
        assert table.attrs["excluded"] == 1

    def test_groups_add_up(self) -> None:
        table = eval_report.aggregate(self.frame(), "signal")
        assert int(table.loc[1:, "queries"].sum()) == 1

    def test_empty_column_still_returns_table(self) -> None:
        table = eval_report.aggregate(self.frame().iloc[0:0])
        assert table.loc[0, "queries"] == 0


class TestTaxonomy:
    def frame(self) -> pd.DataFrame:
        train, validation = make_frames()
        return eval_report.evaluate(build_baseline(train), validation)

    def test_counts_within_denominator(self) -> None:
        errors = eval_report.error_taxonomy(self.frame())
        assert (errors["count"] <= errors["denominator"]).all()
        assert (errors["count"] >= 0).all()

    def test_main_sample_denominator_is_query_count(self) -> None:
        frame = self.frame()
        errors = eval_report.error_taxonomy(frame).set_index("type")
        row = errors.loc["Причина не найдена в top-3"]
        assert row["denominator"] == len(frame)

    def test_p1_row_denominator_counts_p1_queries(self) -> None:
        frame = self.frame()
        errors = eval_report.error_taxonomy(frame).set_index("type")
        assert errors.loc["Причина не найдена у P1", "denominator"] == 1

    def test_taxonomy_is_deterministic(self) -> None:
        frame = self.frame()
        first = eval_report.error_taxonomy(frame)
        second = eval_report.error_taxonomy(frame)
        pd.testing.assert_frame_equal(first, second)

    def test_no_nan_examples(self) -> None:
        errors = eval_report.error_taxonomy(self.frame())
        assert errors["example"].notna().all()


class TestRendering:
    def test_tables_have_no_stray_blank_lines(self) -> None:
        train, validation = make_frames()
        frame = eval_report.evaluate(build_baseline(train), validation)
        errors = eval_report.error_taxonomy(frame)
        summary = eval_report.build_summary(frame, errors, validation, "hash", 1)
        text = eval_report.render_taxonomy(errors, summary)
        assert "|\n\n|" not in text

    def test_summary_alpha_is_baseline_plus_margin(self) -> None:
        train, validation = make_frames()
        frame = eval_report.evaluate(build_baseline(train), validation)
        errors = eval_report.error_taxonomy(frame)
        summary = eval_report.build_summary(frame, errors, validation, "hash", 1)
        assert summary["thresholds"]["alpha"] == pytest.approx(
            summary["ndcg"]["v2"] + eval_report.ALPHA_MARGIN, abs=1e-5
        )

    def test_summary_is_json_serialisable(self) -> None:
        train, validation = make_frames()
        frame = eval_report.evaluate(build_baseline(train), validation)
        errors = eval_report.error_taxonomy(frame)
        summary = eval_report.build_summary(frame, errors, validation, "hash", 1)
        assert json.loads(json.dumps(summary, allow_nan=False))


@requires_data
class TestOnRealData:
    @pytest.fixture(scope="class")
    @classmethod
    def evaluation(cls) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from prepare_features import fit_vectorizer, load_splits, transform

        frames = load_splits()
        vectorizer = fit_vectorizer(frames["train"])
        baseline = RetrievalBaseline(
            vectorizer, frames["train"], transform(vectorizer, frames["train"])
        )
        sample = frames["validation"].head(40)
        return eval_report.evaluate(baseline, sample), frames["validation"], frames["train"]

    def test_no_queries_without_relevant_cards(self, evaluation) -> None:
        frame, _, _ = evaluation
        assert (frame["ideal_v2"] > 0).all()

    def test_ideal_reaches_three_grade_two(self, evaluation) -> None:
        frame, _, _ = evaluation
        assert (frame["ideal_v2"] >= 3).all()

    def test_grade_two_is_reachable(self, evaluation) -> None:
        frame, _, _ = evaluation
        assert frame["grades_v2"].str.contains("2").any()

    def test_manual_case_matches_evaluate(self, evaluation) -> None:
        frame, validation, train = evaluation
        from prepare_features import fit_vectorizer, transform

        with_two = frame[frame["grades_v2"].str.contains("2")]
        if with_two.empty:
            pytest.skip("в выборке нет карточки с оценкой 2")
        query_id = with_two.iloc[0]["incident_id"]
        vectorizer = fit_vectorizer(train)
        baseline = RetrievalBaseline(vectorizer, train, transform(vectorizer, train))
        case = eval_report.manual_case(baseline, frame, validation, query_id)
        row = frame.set_index("incident_id").loc[case["incident_id"]]
        assert case["grades"] == [int(value) for value in row["grades_v2"].split(",")]
        assert case["ndcg"] == pytest.approx(float(row["ndcg_v2"]), abs=1e-5)

    def test_ceiling_is_an_upper_bound(self, evaluation) -> None:
        frame, _, _ = evaluation
        assert (frame["ceiling_3"] >= frame["ndcg_v2"] - 1e-9).all()
        assert (frame["ceiling_50"] <= 1.0).all()

    def test_ceiling_grows_with_pool(self, evaluation) -> None:
        frame, _, _ = evaluation
        pools = [frame[f"ceiling_{pool}"] for pool in eval_report.ORACLE_POOLS]
        for smaller, larger in zip(pools, pools[1:]):
            assert (larger >= smaller - 1e-9).all()

    def test_grade_two_candidate_is_in_train(self, evaluation) -> None:
        _, validation, train = evaluation
        assert set(validation["root_cause"]) <= set(train["root_cause"])


class TestHelpers:
    def test_ranked_ids_matches_search_order(self) -> None:
        train, validation = make_frames()
        baseline = build_baseline(train)
        for _, query in validation.iterrows():
            vector = eval_report.query_vector(baseline, query["description"])
            scores = eval_report.similarity_scores(baseline, vector)
            expected = [hit.incident_id for hit in baseline.search(query["description"]).hits]
            assert eval_report.ranked_ids(baseline, scores, 3) == expected

    def test_ranked_ids_is_empty_for_zero_scores(self) -> None:
        train, _ = make_frames()
        baseline = build_baseline(train)
        scores = pd.Series([0.0] * len(train)).to_numpy()
        assert eval_report.ranked_ids(baseline, scores, 3) == []

    def test_neighbour_signals_without_neighbours_is_zero(self) -> None:
        train, _ = make_frames()
        baseline = build_baseline(train)
        scores = pd.Series([0.0] * len(train)).to_numpy()
        query = pd.Series({"service": "ims", "root_cause": "capacity", "description": ""})
        service_max, cause_max = eval_report.neighbour_signals(baseline, scores, query)
        assert (service_max, cause_max) == (0.0, 0.0)

    def test_neighbour_signals_ignore_same_card(self) -> None:
        train, validation = make_frames()
        baseline = build_baseline(train)
        query = validation.iloc[0]
        vector = eval_report.query_vector(baseline, query["description"])
        scores = eval_report.similarity_scores(baseline, vector)
        service_max, cause_max = eval_report.neighbour_signals(baseline, scores, query)
        assert service_max == pytest.approx(0.0) or cause_max > 0

    def test_empty_text_has_zero_vector(self) -> None:
        train, _ = make_frames()
        baseline = build_baseline(train)
        assert eval_report.query_vector(baseline, "").nnz == 0
        assert eval_report.query_vector(baseline, None).nnz == 0

    def test_manifest_mismatch_raises(self, tmp_path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"vocabulary_sha256": "deadbeef"}), encoding="utf-8")
        with pytest.raises(ValueError):
            eval_report.check_manifest(path, "cafebabe")

    def test_missing_manifest_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            eval_report.check_manifest(tmp_path / "absent.json", "cafebabe")

    def test_manifest_match_passes(self, tmp_path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"vocabulary_sha256": "cafebabe"}), encoding="utf-8")
        assert eval_report.check_manifest(path, "cafebabe") is None

    def test_cause_rank_is_one_when_query_itself_is_best(self) -> None:
        train, _ = make_frames()
        baseline = build_baseline(train)
        row = train.iloc[0]
        vector = eval_report.query_vector(baseline, row["description"])
        scores = eval_report.similarity_scores(baseline, vector)
        assert eval_report.label_rank(baseline, scores, "root_cause", row["root_cause"]) == 1


@requires_data
class TestNoLeak:
    def test_queries_have_no_cards_of_their_event_in_index(self) -> None:
        from prepare_features import load_splits

        frames = load_splits()
        train_events = set(frames["train"]["event_id"])
        assert not train_events & set(frames["validation"]["event_id"])
        assert not train_events & set(frames["test"]["event_id"])
        assert not set(frames["train"]["incident_id"]) & set(frames["validation"]["incident_id"])


class TestIdealBoundaries:
    """Идеал собирается из карточек индекса: сколько двоек и единиц в train."""

    def test_ideal_counts_are_zero_when_nothing_matches(self) -> None:
        train, validation = make_frames()
        counts = eval_report.ideal_counts(build_baseline(train), validation.iloc[1])["v2"]
        assert counts == (0, 0)

    def test_ideal_order_is_twos_then_ones_then_padding(self) -> None:
        assert ideal_grades(2, 1, 3) == [2, 2, 1]
        assert ideal_grades(1, 5, 3) == [2, 1, 1]
        assert ideal_grades(0, 0, 3) == []

    def test_ideal_never_exceeds_k(self) -> None:
        for two, one in [(5, 5), (1, 1), (0, 4), (4, 0)]:
            assert len(ideal_grades(two, one, 3)) <= 3

    @pytest.mark.parametrize("grades", list(itertools.permutations([2, 2, 1])))
    def test_ndcg_with_possible_grades_stays_within_one(self, grades) -> None:
        value = ndcg(list(grades), ideal_grades(2, 1, 3), 3)
        assert 0.0 <= value <= 1.0

    def test_ndcg_is_zero_for_empty_ideal(self) -> None:
        assert ndcg([2, 1, 0], ideal_grades(0, 0, 3), 3) == 0.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_gives_zero_without_exception(self, k: int) -> None:
        assert ndcg([2, 2, 2], ideal_grades(5, 5, 3), k) == 0.0


class TestEmptyAndZeroOutput:
    def test_unmatched_query_has_zero_metrics_not_nan(self) -> None:
        frame, _ = small_evaluation()
        row = frame.set_index("incident_id").loc["Q2"]
        assert row["returned"] == 0
        assert row["ideal_v2"] == 0
        for column in ("ndcg_v2", "recall_v2", "mrr_v2", "ndcg_v1"):
            assert row[column] == 0.0
            assert not math.isnan(float(row[column]))

    def test_taxonomy_on_empty_frame_keeps_computed_rows_zero(self) -> None:
        frame, _ = small_evaluation()
        errors = eval_report.error_taxonomy(frame.iloc[0:0]).set_index("type")
        assert (errors["count"] <= errors["denominator"]).all()
        for kind in ("Причина не найдена в top-3", "Причина не найдена у P1", "Пустая выдача"):
            assert errors.loc[kind, "count"] == 0
            assert errors.loc[kind, "denominator"] == 0
        assert errors["example"].notna().all()


class TestJsonWriters:
    def test_write_json_refuses_nan(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            eval_report.write_json(tmp_path / "bad.json", {"value": float("nan")})

    def test_write_json_keeps_cyrillic_and_trailing_newline(self, tmp_path) -> None:
        path = tmp_path / "good.json"
        eval_report.write_json(path, {"rule": "2 — та же root_cause"})
        text = path.read_text(encoding="utf-8")
        assert "та же root_cause" in text
        assert text.endswith("\n")
        assert json.loads(text) == {"rule": "2 — та же root_cause"}


class TestTaxonomyConsistency:
    def test_p1_denominator_is_p1_count_not_total(self) -> None:
        train, validation = make_frames()
        extra = validation.iloc[[0]].copy()
        extra["incident_id"] = "Q3"
        validation = pd.concat([validation, extra], ignore_index=True)
        frame = eval_report.evaluate(build_baseline(train), validation)
        row = eval_report.error_taxonomy(frame).set_index("type").loc["Причина не найдена у P1"]
        assert row["denominator"] == 2
        assert row["denominator"] != len(frame)

    @pytest.mark.parametrize("share", [0.10, 0.30, 0.50, 0.51, 0.90])
    def test_noise_row_matches_noise_high_band(self, share: float) -> None:
        frame, _ = small_evaluation()
        frame = frame.copy()
        frame["noise_share"] = [0.0, share]
        frame["noise"] = [eval_report.noise_band(value) for value in frame["noise_share"]]
        row = eval_report.error_taxonomy(frame).set_index("type").loc[
            "Сходство top-1 больше чем наполовину на служебных словах"
        ]
        assert row["count"] == int((frame["noise"] == "noise_high").sum())


@requires_artifacts
class TestArtifactConsistency:
    """Опубликованные документы и results/* не должны расходиться."""

    def summary(self) -> dict[str, object]:
        return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

    def test_overall_row_matches_summary(self) -> None:
        summary = self.summary()
        rows = [row for row in table_rows(read_doc("segments.md")) if row[0] == "Все запросы"]
        assert rows
        for row in rows:
            assert int(row[1]) == summary["queries"]
            assert parse_number(row[2]) == pytest.approx(summary["ndcg"]["v2"], abs=5e-4)
            assert parse_number(row[9]) == pytest.approx(summary["ndcg"]["v1"], abs=5e-4)

    def test_alpha_follows_the_declared_rule(self) -> None:
        summary = self.summary()
        assert summary["thresholds"]["alpha"] == pytest.approx(
            summary["ndcg"]["v2"] + eval_report.ALPHA_MARGIN, abs=1e-6
        )
        assert str(eval_report.ALPHA_MARGIN) in summary["thresholds"]["alpha_rule"]

    def test_protective_metrics_match_p1_queries(self) -> None:
        summary = self.summary()
        frame = pd.read_csv(RESULTS / "per_query.csv", dtype={"grades_v2": str})
        p1 = frame[frame["severity"] == "P1"]
        protective = summary["thresholds"]["protective"]
        assert len(p1) == summary["p1"]["queries"]
        assert protective["p1_cause_hit"] == pytest.approx(float(p1["cause_exact_hit"].mean()), abs=1e-6)
        assert protective["p1_ndcg_v2"] == pytest.approx(float(p1["ndcg_v2"].mean()), abs=1e-6)
        assert protective["empty"] == int((frame["returned"] == 0).sum())

    def test_summary_means_match_per_query_csv(self) -> None:
        summary = self.summary()
        frame = pd.read_csv(RESULTS / "per_query.csv", dtype={"grades_v2": str})
        assert summary["queries"] == len(frame)
        assert summary["ndcg"]["v2"] == pytest.approx(float(frame["ndcg_v2"].mean()), abs=1e-6)
        assert summary["recall"]["v2"] == pytest.approx(float(frame["recall_v2"].mean()), abs=1e-6)
        assert summary["mrr"]["v2"] == pytest.approx(float(frame["mrr_v2"].mean()), abs=1e-6)
        assert sum(summary["patterns"]["counts"].values()) == len(frame)
        assert summary["no_relevant_queries"] == int((frame["ideal_v2"] == 0).sum())

    def test_json_files_are_strictly_valid(self) -> None:
        for name in ("summary.json", "manual_case.json"):
            text = (RESULTS / name).read_text(encoding="utf-8")
            payload = json.loads(text, parse_constant=lambda value: pytest.fail(f"{name}: константа {value}"))
            assert payload

    def test_taxonomy_doc_matches_errors_csv(self) -> None:
        errors = pd.read_csv(RESULTS / "errors.csv").set_index("type")
        checked = 0
        for row in table_rows(read_doc("error_taxonomy.md")):
            if len(row) != 6 or row[0] not in errors.index:
                continue
            match = re.match(r"(\d+) из (\d+)", row[1])
            assert match, row[1]
            assert int(match.group(1)) == int(errors.loc[row[0], "count"]), row[0]
            assert int(match.group(2)) == int(errors.loc[row[0], "denominator"]), row[0]
            checked += 1
        assert checked == len(errors)

    def test_manual_case_reproduces_from_its_own_numbers(self) -> None:
        case = json.loads((RESULTS / "manual_case.json").read_text(encoding="utf-8"))
        grades = case["grades"]
        assert case["dcg"] == pytest.approx(dcg(grades, 3), abs=1e-5)
        assert case["idcg"] == pytest.approx(dcg(case["ideal_grades"], 3), abs=1e-5)
        assert case["ndcg"] == pytest.approx(ndcg(grades, case["ideal_grades"], 3), abs=1e-5)
        assert [hit["grade"] for hit in case["hits"]] == grades
        assert [hit["rank"] for hit in case["hits"]] == list(range(1, len(grades) + 1))
        assert grades[:3] == [2, 1, 0]
        assert case["ideal_grades"][:3] == [2, 2, 2]
        assert case["counts"]["grade_two"] >= 3

    def test_manual_calculation_doc_matches_json(self) -> None:
        case = json.loads((RESULTS / "manual_case.json").read_text(encoding="utf-8"))
        text = read_doc("manual_calculation.md")
        assert case["incident_id"] in text
        match = re.search(r"NDCG@3 = (\d+,\d+) / (\d+,\d+) = (\d+,\d+)", text)
        assert match
        assert parse_number(match.group(1)) == pytest.approx(case["dcg"], abs=1e-5)
        assert parse_number(match.group(2)) == pytest.approx(case["idcg"], abs=1e-5)
        assert parse_number(match.group(3)) == pytest.approx(case["ndcg"], abs=1e-3)

    def test_generated_tables_have_no_placeholder_cells(self) -> None:
        banned = {"nan", "-nan", "inf", "-inf", "none", "nat"}
        for name in GENERATED_DOCS:
            text = read_doc(name)
            for row in table_rows(text):
                for cell in row:
                    assert cell.lower() not in banned, f"{name}: {cell}"
            assert "|\n\n|" not in text


@requires_artifacts
class TestRegressionGuards:
    """Проверки дефектов, найденных ревью и тестами: раньше падали, теперь держат регрессию."""

    def test_taxonomy_names_every_pattern(self) -> None:
        summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        bullet = next(line for line in read_doc("error_taxonomy.md").splitlines() if "Режимы выдачи" in line)
        for pattern in ("2,2,2", "1,1,1", "0,0,0"):
            if pattern in summary["patterns"]["counts"]:
                assert f"«{pattern}» — {summary['patterns']['counts'][pattern]}" in bullet, bullet
                assert f"«{pattern},1,1»" not in bullet

    def test_manual_swap_sentence_matches_the_operation(self) -> None:
        case = manual_payload([2, 1, 0])
        text = eval_report.render_manual(case)
        match = re.search(r"полезность (\d+,\d+)", text)
        assert match
        swapped = [case["grades"][1], case["grades"][0], case["grades"][2]]
        assert parse_number(match.group(1)) == pytest.approx(dcg(swapped, 3), abs=1e-4)

    def test_residual_pattern_row_gets_a_number(self) -> None:
        frame, validation = small_evaluation()
        covered = frame.copy()
        covered["grades_v2"] = "0,0,0"
        summary = eval_report.build_summary(
            covered, eval_report.error_taxonomy(covered), validation, "hash", 0
        )
        text = eval_report.render_segments(covered, summary)
        row = next(row for row in table_rows(text) if row[0] == "прочие")
        assert row[2] == "0"
        assert row[4] == "—" or not math.isnan(parse_number(row[4]))

    def test_single_scenario_spread_gets_a_number(self) -> None:
        frame, validation = small_evaluation()
        single = frame.copy()
        single["scenario_id"] = "SC-ONLY"
        summary = eval_report.build_summary(
            single, eval_report.error_taxonomy(single), validation, "hash", 0
        )
        assert not re.search(r"\bnan\b", eval_report.render_segments(single, summary).lower())

    def test_query_without_relevant_cards_leaves_the_mean(self) -> None:
        frame, _ = small_evaluation()
        assert (frame["ideal_v2"] == 0).any()
        expected = float(frame.loc[frame["ideal_v2"] > 0, "ndcg_v2"].mean())
        assert eval_report.aggregate(frame).loc[0, "ndcg_v2"] == pytest.approx(expected)

    def test_run_writes_documents_next_to_results(self, tmp_path, monkeypatch) -> None:
        train, validation = make_frames()
        monkeypatch.setattr(
            eval_report, "load_splits", lambda: {"train": train, "validation": validation}
        )
        frame = eval_report.evaluate(build_baseline(train), validation)
        frame.loc[frame.index[0], "grades_v2"] = "2,1,0"
        monkeypatch.setattr(eval_report, "evaluate", lambda baseline, queries: frame)
        monkeypatch.setattr(eval_report, "check_manifest", lambda path, vocabulary: None)
        target = tmp_path / "out"
        docs = tmp_path / "docs"
        eval_report.run(target, docs_dir=docs)
        assert (docs / "segments.md").exists()
        assert not (PROJECT_ROOT / "docs" / "lab8" / "results" / "out").exists()

    def test_build_summary_on_empty_frame_raises_value_error(self) -> None:
        frame, validation = small_evaluation()
        empty = frame.iloc[0:0]
        with pytest.raises(ValueError):
            eval_report.build_summary(empty, eval_report.error_taxonomy(empty), validation, "hash", 0)

    def test_render_segments_on_empty_frame_raises_value_error(self) -> None:
        frame, validation = small_evaluation()
        empty = frame.iloc[0:0]
        summary = eval_report.build_summary(frame, eval_report.error_taxonomy(frame), validation, "hash", 0)
        with pytest.raises(ValueError):
            eval_report.render_segments(empty, summary)

    def test_unknown_query_id_raises_value_error(self) -> None:
        train, validation = make_frames()
        baseline = build_baseline(train)
        frame = eval_report.evaluate(baseline, validation)
        with pytest.raises(ValueError):
            eval_report.manual_case(baseline, frame, validation, "I999999")

    def test_gap_error_example_comes_from_counted_rows(self) -> None:
        frame, _ = small_evaluation()
        frame = frame.copy()
        frame["gap"] = [-0.005, -0.5]
        frame["signal"] = [eval_report.signal_band(value) for value in frame["gap"]]
        row = eval_report.error_taxonomy(frame).set_index("type").loc["Причина дальше, чем услуга"]
        assert row["count"] == 1
        example_id = str(row["example"]).split()[0]
        assert float(frame.set_index("incident_id").loc[example_id, "gap"]) < -eval_report.GAP_TIE


@requires_artifacts
class TestRunDeterminism:
    def test_two_runs_are_byte_identical(self, tmp_path, monkeypatch) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        docs_first = tmp_path / "docs_first"
        assert eval_report.run(first, docs_dir=docs_first) == 0
        docs_second = tmp_path / "docs_second"
        assert eval_report.run(second, docs_dir=docs_second) == 0

        names = sorted(path.name for path in first.iterdir())
        assert names == sorted(path.name for path in second.iterdir())
        for name in names:
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

        docs = sorted(path.name for path in (tmp_path / "docs_first").iterdir())
        assert docs == sorted(GENERATED_DOCS)
        for name in docs:
            assert (tmp_path / "docs_first" / name).read_bytes() == (
                tmp_path / "docs_second" / name
            ).read_bytes(), name

