"""Baseline проектной задачи (ЛР 7): retrieval top-3 и частотный класс.

Условия зафиксированы до первого запуска и записываются в `settings.json`:
данные 1.1, индекс только из train, запросы только из validation, k = 3,
признак — единственное поле `description`, оценки релевантности считаются по
метками, а не по тексту. Модель детерминирована: случайность не используется,
поле `seed` в настройках пустое.

Запуск из корня репозитория:
    python src/triage_baseline.py run
    python src/triage_baseline.py search --query "рост таймаутов Diameter"
    python src/triage_baseline.py search --query-id I000421
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, recall_score

from prepare_features import (
    SPLIT_FILES,
    TEXT_COLUMN,
    fit_vectorizer,
    load_splits,
    transform,
    vocabulary_hash,
)

BASELINE_VERSION = "1.0"
TOP_K = 3
TARGETS = ("severity", "service", "root_cause")
RESULTS_DIR = Path("docs/lab7/results")
CASES_LIMIT = 10
MANIFEST_PATH = Path("docs/lab6/features_manifest.json")
LANGUAGE_COLUMN = "language"

# Правило релевантности опубликовано до первого запуска и не подбирается:
# 2 — та же услуга и та же причина, 1 — совпало что-то одно, 0 — ничего.
GRADING_RULE = (
    "2 — совпали service и root_cause; 1 — совпал service или root_cause; "
    "0 — не совпало ни то, ни другое"
)
# Сегмент запроса — доля содержательных слов, найденных в словаре train. Содержательным
# считается слово без цифр, которое встречается не более чем в половине карточек train:
# `ci`, `host` и «объект» есть почти в каждом описании и о знакомстве запроса не говорят.
# Границы 50 % и 80 % круглые; метрики по запросам от разбиения не зависят, оно показывает,
# где baseline ломается.
SEGMENT_RULE = (
    "доля уникальных содержательных слов (без цифр, встречаются не более чем в половине "
    "карточек train), найденных в словаре train: 0 — unknown_all, до 50 % — coverage_low, "
    "до 80 % — coverage_mid, выше — coverage_high"
)
SEGMENT_THRESHOLDS = (0.5, 0.8)
BOILERPLATE_DF = 0.5
# Механизмы отказа baseline; номера используются в cases.md и в README.
FAILURE_MECHANISMS = {
    1: "словоформы: «таймаут» и «таймауты» — разные признаки",
    2: "синонимы и перифраз не пересекаются со словарём train",
    3: "общие слова причины и последствия весят больше названия услуги",
    4: "одно сходство отвечает и за услугу, и за причину",
}
METRIC_DEFINITIONS = {
    "ndcg_at_k": "DCG с gain 2^оценка - 1 и дисконтом 1/log2(позиция + 1), делённый на идеальный DCG того же запроса",
    "recall_at_k": "доля запросов, где в top-k есть хотя бы одна карточка с оценкой не ниже 1 (попадание, а не доля найденных релевантных)",
    "full_hit_at_k": "доля запросов, где в top-k есть хотя бы одна карточка с оценкой 2",
    "mrr_at_k": "обратная позиция первой карточки с оценкой не ниже 1, 0 если её нет в top-k",
    "f1_macro": "среднее F1 по полному набору меток train и validation; класс без объектов получает 0",
}


@dataclass(frozen=True)
class Hit:
    """Одна карточка в выдаче: идентификатор и косинусное сходство."""

    incident_id: str
    score: float


@dataclass(frozen=True)
class SearchResult:
    """Выдача по запросу вместе с покрытием словаря, из которого считается сегмент."""

    hits: tuple[Hit, ...]
    known_tokens: int
    total_tokens: int

    @property
    def coverage(self) -> float:
        """Доля слов запроса без цифр, известных словарю train."""
        if self.total_tokens == 0:
            return 0.0
        return self.known_tokens / self.total_tokens

    @property
    def segment(self) -> str:
        """Сегмент запроса по покрытию словаря train. Границы зафиксированы в SEGMENT_RULE."""
        coverage = self.coverage
        if self.total_tokens == 0 or self.known_tokens == 0:
            return "unknown_all"
        if coverage <= SEGMENT_THRESHOLDS[0]:
            return "coverage_low"
        if coverage <= SEGMENT_THRESHOLDS[1]:
            return "coverage_mid"
        return "coverage_high"


def sha256_file(path: Path) -> str:
    """Отпечаток файла данных: по нему видно, что данные между запусками не менялись."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relevance_grade(service: str, cause: str, candidate: pd.Series) -> int:
    """Уровень релевантности кандидата для запроса с указанными метками."""
    same_service = candidate["service"] == service
    same_cause = candidate["root_cause"] == cause
    if same_service and same_cause:
        return 2
    if same_service or same_cause:
        return 1
    return 0


def dcg(grades: list[int], k: int) -> float:
    """Сумма gain / log2(позиция + 1) по первым k позициям; k не меньше нуля."""
    return sum(
        (2**grade - 1) / math.log2(position + 2)
        for position, grade in enumerate(grades[: max(k, 0)])
    )


def ndcg(grades: list[int], ideal: list[int], k: int) -> float:
    """NDCG первых k позиций; при пустом идеале метрика равна нулю."""
    best = dcg(ideal, k)
    if best == 0:
        return 0.0
    return dcg(grades, k) / best


def ideal_grades(grade_two: int, grade_one: int, k: int) -> list[int]:
    """Идеальный порядок: полностью релевантные карточки впереди частичных."""
    grades = [2] * grade_two + [1] * grade_one
    return grades[: max(k, 0)]


class RetrievalBaseline:
    """Поиск по train: TF-IDF из ЛР 6 и косинусное сходство, без обучения ранжировщика."""

    def __init__(self, vectorizer, train: pd.DataFrame, matrix) -> None:
        if matrix.shape[0] != len(train):
            raise ValueError(
                f"матрица {matrix.shape[0]} строк не соответствует train из {len(train)} карточек"
            )
        self.vectorizer = vectorizer
        self.train = train.reset_index(drop=True)
        self.matrix = matrix
        self._ids = self.train["incident_id"].to_numpy()
        self._services = self.train["service"].to_numpy()
        self._causes = self.train["root_cause"].to_numpy()
        self._positions = {incident_id: index for index, incident_id in enumerate(self._ids)}
        self._unigram_vocabulary = {
            term for term in vectorizer.get_feature_names_out() if " " not in term
        }
        self._analyzer = vectorizer.build_analyzer()
        # Служебные слова: встречаются больше чем в половине карточек train, содержания не несут.
        document_frequency = np.asarray((self.matrix > 0).sum(axis=0)).ravel()
        shares = document_frequency / max(len(self.train), 1)
        features = vectorizer.get_feature_names_out()
        self._features = features
        self._term_df = {
            term: float(share) for term, share in zip(features, shares) if " " not in term
        }
        self._boilerplate = {
            term for term, share in self._term_df.items() if share > BOILERPLATE_DF
        }

    def explain_similarity(self, text: str, card: pd.Series, limit: int = 5) -> list[str]:
        """Слова с наибольшим вкладом в косинусное сходство с карточкой.

        Разложение скалярного произведения по словам: видно, на чём держится похожесть.
        Служебные слова train (`ci`, `host`) в список не попадают.
        """
        if not isinstance(text, str) or card["incident_id"] not in self._positions:
            return []
        query_vector = self.vectorizer.transform([text])
        card_row = self.matrix[self._positions[card["incident_id"]]]
        product = query_vector.multiply(card_row).tocoo()
        pairs = [
            (float(value), self._features[index])
            for value, index in zip(product.data, product.col)
            if " " not in self._features[index]
        ]
        return [
            term
            for _, term in sorted(pairs, key=lambda pair: (-pair[0], pair[1]))
            if term not in self._boilerplate
        ][:limit]

    def content_tokens(self, text: str) -> set[str]:
        """Уникальные слова запроса без цифр и без служебных слов из train."""
        if not isinstance(text, str):
            return set()
        return {
            term
            for term in self._analyzer(text)
            if " " not in term
            and term not in self._boilerplate
            and not any(character.isdigit() for character in term)
        }

    def search(self, text: str, k: int = TOP_K) -> SearchResult:
        """Top-k карточек с ненулевым сходством; при равных оценках порядок по ID."""
        tokens = self.content_tokens(text)
        known = len(tokens & self._unigram_vocabulary)
        info = {"known_tokens": known, "total_tokens": len(tokens)}
        if k <= 0:
            return SearchResult(hits=(), **info)

        vector = self.vectorizer.transform([text if isinstance(text, str) else ""])
        if vector.nnz == 0:
            return SearchResult(hits=(), **info)

        scores = np.asarray((vector @ self.matrix.T).toarray()).ravel()
        candidates = np.flatnonzero(scores > 0)
        if candidates.size == 0:
            return SearchResult(hits=(), **info)

        order = np.lexsort((self._ids[candidates], -scores[candidates]))
        picked = candidates[order[:k]]
        hits = tuple(Hit(self._ids[index], float(scores[index])) for index in picked)
        return SearchResult(hits=hits, **info)

    def labels(self, incident_id: str) -> pd.Series:
        """Строка train по идентификатору: нужна для оценки релевантности и для выдачи."""
        return self.train.iloc[self._positions[incident_id]]

    def relevant_counts(self, service: str, cause: str) -> tuple[int, int]:
        """Сколько в индексе карточек с оценкой 2 и с оценкой 1 для этих меток."""
        same_service = self._services == service
        same_cause = self._causes == cause
        return int((same_service & same_cause).sum()), int((same_service ^ same_cause).sum())


def evaluate_retrieval(baseline: RetrievalBaseline, queries: pd.DataFrame) -> pd.DataFrame:
    """Считает метрики по каждому запросу; агрегаты выводятся из этой таблицы."""
    rows = []
    for _, query in queries.iterrows():
        result = baseline.search(query[TEXT_COLUMN])
        candidates = [baseline.labels(hit.incident_id) for hit in result.hits]
        grades = [
            relevance_grade(query["service"], query["root_cause"], row) for row in candidates
        ]
        grade_two, grade_one = baseline.relevant_counts(query["service"], query["root_cause"])
        ideal = ideal_grades(grade_two, grade_one, TOP_K)
        first_relevant = next(
            (rank for rank, grade in enumerate(grades, start=1) if grade >= 1), None
        )

        rows.append(
            {
                "incident_id": query["incident_id"],
                "scenario_id": query["scenario_id"],
                "language": query[LANGUAGE_COLUMN],
                "service": query["service"],
                "root_cause": query["root_cause"],
                "severity": query["severity"],
                "segment": result.segment,
                "coverage": round(result.coverage, 6),
                "known_tokens": result.known_tokens,
                "total_tokens": result.total_tokens,
                "returned": len(result.hits),
                "top1_id": result.hits[0].incident_id if result.hits else "",
                "top1_score": round(result.hits[0].score, 6) if result.hits else 0.0,
                "top1_grade": grades[0] if grades else 0,
                "service_hit": int(any(row["service"] == query["service"] for row in candidates)),
                "cause_hit": int(
                    any(row["root_cause"] == query["root_cause"] for row in candidates)
                ),
                "ndcg": round(ndcg(grades, ideal, TOP_K), 6),
                "recall": int(any(grade >= 1 for grade in grades)),
                "full_hit": int(any(grade == 2 for grade in grades)),
                "mrr": round(1 / first_relevant, 6) if first_relevant else 0.0,
            }
        )
    return pd.DataFrame(rows)


def aggregate_retrieval(per_query: pd.DataFrame) -> dict[str, object]:
    """Средние метрики по всему набору и по разрезам: покрытие, язык, сценарий."""
    if per_query.empty or "coverage" not in per_query.columns:
        per_query = per_query.iloc[0:0].reindex(
            columns=[
                "ndcg",
                "recall",
                "full_hit",
                "mrr",
                "service_hit",
                "cause_hit",
                "returned",
                "coverage",
                "segment",
                "language",
                "scenario_id",
            ]
        )

    def summarise(frame: pd.DataFrame) -> dict[str, object]:
        if len(frame) == 0:
            return {
                "queries": 0,
                "ndcg_at_k": 0.0,
                "recall_at_k": 0.0,
                "full_hit_at_k": 0.0,
                "mrr_at_k": 0.0,
                "service_hit_at_k": 0.0,
                "cause_hit_at_k": 0.0,
                "both_labels_at_k": 0.0,
                "empty_results": 0,
            }
        return {
            "queries": int(len(frame)),
            "ndcg_at_k": round(float(frame["ndcg"].mean()), 6),
            "recall_at_k": round(float(frame["recall"].mean()), 6),
            "full_hit_at_k": round(float(frame["full_hit"].mean()), 6),
            "mrr_at_k": round(float(frame["mrr"].mean()), 6),
            "service_hit_at_k": round(float(frame["service_hit"].mean()), 6),
            "cause_hit_at_k": round(float(frame["cause_hit"].mean()), 6),
            "both_labels_at_k": round(float((frame["service_hit"] & frame["cause_hit"]).mean()), 6),
            "empty_results": int((frame["returned"] == 0).sum()),
        }

    coverage = {
        "min": round(float(per_query["coverage"].min()), 6) if len(per_query) else 0.0,
        "median": round(float(per_query["coverage"].median()), 6) if len(per_query) else 0.0,
        "max": round(float(per_query["coverage"].max()), 6) if len(per_query) else 0.0,
        "fully_known_queries": int((per_query["coverage"] == 1.0).sum()) if len(per_query) else 0,
    }
    return {
        "all": summarise(per_query),
        "by_segment": {
            segment: summarise(per_query[per_query["segment"] == segment])
            for segment in ("coverage_low", "coverage_mid", "coverage_high", "unknown_all")
        },
        "by_language": {
            language: summarise(per_query[per_query["language"] == language])
            for language in sorted(per_query["language"].unique())
        },
        "by_scenario": {
            scenario: summarise(per_query[per_query["scenario_id"] == scenario])
            for scenario in sorted(per_query["scenario_id"].unique())
        },
        "coverage": coverage,
    }


def label_structure(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    """Проверяет, достижима ли полная релевантность: общие пары меток индекса и запросов."""
    index_pairs = {(service, cause) for service, cause in zip(train["service"], train["root_cause"])}
    query_pairs = {
        (service, cause) for service, cause in zip(validation["service"], validation["root_cause"])
    }
    shared = index_pairs & query_pairs
    return {
        "index_service_cause_pairs": len(index_pairs),
        "query_service_cause_pairs": len(query_pairs),
        "shared_service_cause_pairs": len(shared),
        "grade_two_reachable": bool(shared),
        "note": (
            "если общих пар нет, полностью релевантных кандидатов в индексе не существует "
            "и идеальная выдача состоит из карточек с оценкой 1"
        ),
    }


def majority_class(train: pd.DataFrame, target: str) -> str:
    """Самый частый класс train; при равенстве частот побеждает меньшая метка по алфавиту."""
    counts = train[target].value_counts()
    if counts.empty:
        raise ValueError(f"в train нет ни одного значения {target}")
    best = counts.max()
    return sorted(counts[counts == best].index)[0]


def evaluate_classification(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    """Частотный класс как baseline триажа: одна константа на весь validation."""
    reports = {}
    for target in TARGETS:
        prediction = majority_class(train, target)
        labels = sorted(set(train[target]) | set(validation[target]))
        predicted = np.full(len(validation), prediction)
        per_class = recall_score(
            validation[target], predicted, labels=labels, average=None, zero_division=0
        )
        reports[target] = {
            "majority_class": prediction,
            "train_share": round(float((train[target] == prediction).mean()), 6),
            "accuracy": round(float(accuracy_score(validation[target], predicted)), 6),
            "f1_macro": round(
                float(
                    f1_score(
                        validation[target], predicted, labels=labels, average="macro", zero_division=0
                    )
                ),
                6,
            ),
            "recall_per_class": {
                label: round(float(value), 6) for label, value in zip(labels, per_class)
            },
            "support": {label: int((validation[target] == label).sum()) for label in labels},
            "distinct_classes": len(labels),
            "labels": labels,
        }
    return reports


def pick_cases(per_query: pd.DataFrame, metric: str, threshold: float, limit: int) -> list[str]:
    """Детерминированная подборка: по кругу из сценариев, чтобы не брать один шаблон."""
    pool = per_query[per_query[metric] == threshold].sort_values(["scenario_id", "incident_id"])
    groups = {
        scenario: frame["incident_id"].tolist() for scenario, frame in pool.groupby("scenario_id")
    }
    picked: list[str] = []
    while len(picked) < limit and any(groups.values()):
        for scenario in sorted(groups):
            if groups[scenario] and len(picked) < limit:
                picked.append(groups[scenario].pop(0))
    return picked


def diagnose_case(query: pd.Series, hits: list[dict[str, object]], grades: list[int]) -> dict[str, object]:
    """Объясняет ответ проверяемым признаком и номером механизма отказа."""
    same_service = sum(1 for hit in hits if hit["service"] == query["service"])
    same_cause = sum(1 for hit in hits if hit["root_cause"] == query["root_cause"])
    counts = f"в top-3 карточек с той же услугой — {same_service}, с той же причиной — {same_cause}"
    if not hits:
        return {
            "mechanism": 2,
            "text": "выдача пуста: ни одно содержательное слово запроса не входит в словарь train",
        }
    if grades[0] == 0 and any(grade >= 1 for grade in grades):
        rank = next(index for index, grade in enumerate(grades, start=1) if grade >= 1)
        return {
            "mechanism": 4,
            "text": (
                "первая карточка не совпала ни по услуге, ни по причине, "
                f"релевантная стоит на {rank}-м месте; {counts}"
            ),
        }
    if all(grade == 0 for grade in grades):
        return {
            "mechanism": 3,
            "text": (
                "ни одна карточка не совпала по меткам, сходство дали общие слова "
                f"формулировки; {counts}"
            ),
        }
    if grades[0] == 2:
        return {"mechanism": None, "text": f"первая карточка совпала и по услуге, и по причине; {counts}"}
    matched = "услуга" if hits[0]["service"] == query["service"] else "причина"
    return {
        "mechanism": 3,
        "text": f"первая карточка совпала по метке «{matched}», вторая метка не совпала; {counts}",
    }


def build_cases(
    baseline: RetrievalBaseline, queries: pd.DataFrame, per_query: pd.DataFrame, limit: int
) -> dict[str, list[dict[str, object]]]:
    """Подборка удачных и проблемных ответов для отчёта и машинного файла."""
    by_id = queries.set_index("incident_id")
    per_query_by_id = per_query.set_index("incident_id")
    cases: dict[str, list[dict[str, object]]] = {
        "success": [],
        "failure": [],
    }
    for kind, threshold in (("success", 1.0), ("failure", 0.0)):
        for incident_id in pick_cases(per_query, "ndcg", threshold, limit):
            query = by_id.loc[incident_id]
            result = baseline.search(query[TEXT_COLUMN])
            hits = []
            candidate_rows = []
            for rank, hit in enumerate(result.hits, start=1):
                row = baseline.labels(hit.incident_id)
                candidate_rows.append(row)
                hits.append(
                    {
                        "rank": rank,
                        "incident_id": hit.incident_id,
                        "score": round(hit.score, 6),
                        "grade": relevance_grade(query["service"], query["root_cause"], row),
                        "service": row["service"],
                        "root_cause": row["root_cause"],
                        "severity": row["severity"],
                    }
                )
            grades = [hit["grade"] for hit in hits]
            terms = (
                baseline.explain_similarity(query[TEXT_COLUMN], candidate_rows[0])
                if candidate_rows
                else []
            )
            diagnosis = diagnose_case(query, hits, grades)
            cases[kind].append(
                {
                    "incident_id": incident_id,
                    "query": {
                        "description": query[TEXT_COLUMN],
                        "service": query["service"],
                        "root_cause": query["root_cause"],
                        "severity": query["severity"],
                        "language": query[LANGUAGE_COLUMN],
                        "segment": per_query_by_id.loc[incident_id, "segment"],
                    },
                    "hits": hits,
                    "ndcg": float(per_query_by_id.loc[incident_id, "ndcg"]),
                    "diagnosis": diagnosis["text"],
                    "mechanism": diagnosis["mechanism"],
                    "mechanism_text": FAILURE_MECHANISMS.get(diagnosis["mechanism"] or 0, ""),
                    "shared_terms": terms,
                }
            )
    return cases


def format_value(value: float, digits: int = 3) -> str:
    """Русская запись числа: десятичная запятая."""
    return f"{value:.{digits}f}".replace(".", ",")


def write_metrics_table(path: Path, retrieval: dict[str, object], classification: dict[str, object]) -> None:
    """Собирает таблицу метрик; числа берутся только из текущего запуска."""
    lines = [
        "# Метрики baseline",
        "",
        "Файл собран командой `python src/triage_baseline.py run`; настройки прогона — "
        "`results/settings.json`, метрики по каждому запросу — `results/retrieval_per_query.csv`.",
        "",
        "## Поиск похожих инцидентов",
        "",
        "| Сегмент запроса | Запросов | NDCG@3 | Recall@3, оценка ≥ 1 | Оценка 2 в top-3 | MRR@3 | Пустых выдач |",
        "|---|---|---|---|---|---|---|",
    ]
    titles = {
        "all": "Все запросы",
        "coverage_low": "Покрытие словаря до 50 %",
        "coverage_mid": "Покрытие словаря 50–80 %",
        "coverage_high": "Покрытие словаря выше 80 %",
        "unknown_all": "Ни одного слова: нулевой вектор",
    }

    def row(title: str, block: dict[str, object]) -> str:
        return (
            f"| {title} | {block['queries']} | {format_value(block['ndcg_at_k'])} | "
            f"{format_value(block['recall_at_k'])} | {format_value(block['full_hit_at_k'])} | "
            f"{format_value(block['mrr_at_k'])} | {block['empty_results']} |"
        )

    lines.append(row(titles["all"], retrieval["all"]))
    for segment, title in titles.items():
        if segment == "all":
            continue
        lines.append(row(title, retrieval["by_segment"][segment]))

    lines += [
        "",
        "Recall@3 здесь — доля запросов с попаданием в top-3, а не доля найденных "
        "релевантных карточек.",
        "",
        "## Разрез по сценариям",
        "",
        "Запросы validation делятся на несколько сценариев, внутри сценария карточки почти "
        "одинаковы. Поэтому строки сценариев показывают разброс лучше, чем одно среднее.",
        "",
        "| Сценарий | Запросов | NDCG@3 | Recall@3 |",
        "|---|---|---|---|",
    ]
    for scenario, block in retrieval["by_scenario"].items():
        lines.append(
            f"| {scenario} | {block['queries']} | {format_value(block['ndcg_at_k'])} | "
            f"{format_value(block['recall_at_k'])} |"
        )

    lines += [
        "",
        "## Разрез по языку запроса",
        "",
        "| Язык | Запросов | NDCG@3 | Recall@3 | MRR@3 |",
        "|---|---|---|---|---|",
    ]
    for language, block in retrieval["by_language"].items():
        lines.append(
            f"| {language} | {block['queries']} | {format_value(block['ndcg_at_k'])} | "
            f"{format_value(block['recall_at_k'])} | {format_value(block['mrr_at_k'])} |"
        )

    structure = retrieval["structure"]
    if structure["grade_two_reachable"]:
        footnote = (
            f"Общих пар «услуга + причина» у индекса и запросов "
            f"{structure['shared_service_cause_pairs']}, поэтому оценка 2 достижима."
        )
    else:
        footnote = (
            "Общих пар «услуга + причина» у индекса и запросов нет "
            f"({structure['index_service_cause_pairs']} пар в индексе, "
            f"{structure['query_service_cause_pairs']} у запросов), поэтому карточек с оценкой 2 "
            "в индексе не существует: NDCG@3 делится на идеал из трёх оценок 1, а столбец "
            "«Оценка 2 в top-3» равен нулю по построению, а не из-за качества поиска."
        )
    lines += [
        "",
        "## Что именно совпало в выдаче",
        "",
        "| Сегмент запроса | Запросов | Услуга в top-3 | Причина в top-3 | Обе метки в top-3 |",
        "|---|---|---|---|---|",
    ]
    for segment, title in titles.items():
        block = retrieval["all"] if segment == "all" else retrieval["by_segment"][segment]
        lines.append(
            f"| {title} | {block['queries']} | {format_value(block['service_hit_at_k'])} | "
            f"{format_value(block['cause_hit_at_k'])} | {format_value(block['both_labels_at_k'])} |"
        )
    lines += ["", footnote, ""]
    coverage = retrieval["coverage"]
    lines.append(
        "Покрытие словаря по запросам: минимум "
        f"{format_value(coverage['min'])}, медиана {format_value(coverage['median'])}, "
        f"максимум {format_value(coverage['max'])}; запросов, где все содержательные слова "
        f"знакомы словарю, {coverage['fully_known_queries']}."
    )

    absent = [
        f"{label} ({target})"
        for target, report in classification.items()
        for label, support in report["support"].items()
        if support == 0
    ]
    macro_note = (
        "F1-macro — среднее по полному набору меток train и validation; класс без объектов "
        "в validation получает 0 и уменьшает среднее"
    )
    macro_note += (
        f". Таких классов: {', '.join(absent)}."
        if absent
        else ". Классов без объектов в validation нет."
    )
    macro_note += " В ЛР 8 сравнение идёт по тому же правилу."
    lines += [
        "",
        "## Классификация: самый частый класс",
        "",
        macro_note,
        "",
        "| Цель | Класс baseline | Доля класса в train | Accuracy | F1-macro |",
        "|---|---|---|---|---|",
    ]
    for target, report in classification.items():
        lines.append(
            f"| {target} | {report['majority_class']} | {format_value(report['train_share'])} | "
            f"{format_value(report['accuracy'])} | {format_value(report['f1_macro'])} |"
        )

    lines += ["", "### Полнота по классам", "", "| Цель | Класс | Объектов | Recall |", "|---|---|---|---|"]
    for target, report in classification.items():
        for label, value in report["recall_per_class"].items():
            lines.append(
                f"| {target} | {label} | {report['support'][label]} | {format_value(value)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def format_terms(terms: list[str]) -> str:
    """Слова в кавычках через запятую; пустой список — прочерк."""
    if not terms:
        return "—"
    return ", ".join(f"«{term}»" for term in terms)


def write_cases(path: Path, cases: dict[str, list[dict[str, object]]]) -> None:
    """Подборка случаев: запрос, выдача с оценками и проверяемое объяснение ответа."""
    lines = [
        "# Подборка удачных и проблемных ответов",
        "",
        "Файл собран командой `python src/triage_baseline.py run`. Подборка детерминирована: "
        "успех — NDCG@3 равен 1, провал — NDCG@3 равен 0, карточки берутся по кругу из сценариев.",
        "",
        "Успех по правилу означает, что все три позиции заняли карточки с оценкой не ниже 1, "
        "то есть совпала услуга или причина. Совпадения обеих меток в индексе нет, поэтому "
        "успешный ответ не равен «нашлась та же причина и та же услуга».",
        "",
        "«Слова с вкладом» — разложение косинусного сходства по словам: видно, на чём "
        "держится похожесть. «Причина» перечисляет совпадения меток в выдаче. "
        "Механизмы отказа перечислены в `README.md`, номера совпадают.",
        "",
    ]
    counts: dict[int, int] = {}
    for entries in cases.values():
        for entry in entries:
            if entry["mechanism"]:
                counts[entry["mechanism"]] = counts.get(entry["mechanism"], 0) + 1
    present = ", ".join(f"{number} — {count}" for number, count in sorted(counts.items())) or "—"
    missing = [str(number) for number in FAILURE_MECHANISMS if number not in counts]
    lines += [
        f"Механизмы в подборке: {present}."
        + (f" Механизмы {', '.join(missing)} не встретились." if missing else ""),
        "",
    ]
    for kind, title in (("success", "Успешные ответы"), ("failure", "Проблемные ответы")):
        entries = cases[kind]
        lines += [f"## {title} ({len(entries)})", ""]
        for entry in entries:
            query = entry["query"]
            lines += [
                f"### {entry['incident_id']}, {query['service']} / {query['root_cause']} / "
                f"{query['severity']}, сегмент {query['segment']}",
                "",
                f"> {query['description']}",
                "",
                "| Ранг | Карточка | Сходство | Оценка | Услуга | Причина | Серьёзность |",
                "|---|---|---|---|---|---|---|",
            ]
            if not entry["hits"]:
                lines.append("| — | выдач нет | 0 | — | — | — | — |")
            for hit in entry["hits"]:
                lines.append(
                    f"| {hit['rank']} | {hit['incident_id']} | {format_value(hit['score'])} | "
                    f"{hit['grade']} | {hit['service']} | {hit['root_cause']} | {hit['severity']} |"
                )
            mechanism = (
                f"{entry['mechanism']} — {entry['mechanism_text']}"
                if entry["mechanism"]
                else "—"
            )
            lines += [
                "",
                f"Причина: {entry['diagnosis']}",
                f"Слова с наибольшим вкладом в сходство: {format_terms(entry['shared_terms'])}",
                f"Механизм отказа: {mechanism}",
                "",
            ]
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def dataset_version_from_paths(paths: dict[str, str]) -> str:
    """Версия набора из имени файла: incidents_train_v1.1.csv → 1.1."""
    match = re.search(r"_v(\d+\.\d+)\.csv$", str(paths.get("train", "")))
    return match.group(1) if match else "unknown"


def build_settings(
    baseline: RetrievalBaseline, paths: dict[str, str] | None = None
) -> dict[str, object]:
    """Настройки прогона: данные, параметры, окружение, команды, отпечатки."""
    paths = paths or SPLIT_FILES
    manifest_hash = None
    if MANIFEST_PATH.exists():
        try:
            manifest_hash = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get(
                "vocabulary_sha256"
            )
        except (OSError, json.JSONDecodeError):
            manifest_hash = None
    return {
        "baseline_version": BASELINE_VERSION,
        "dataset_version": dataset_version_from_paths(paths),
        "index_split": "train",
        "query_split": "validation",
        "index_size": int(len(baseline.train)),
        "top_k": TOP_K,
        "text_column": TEXT_COLUMN,
        "grading_rule": GRADING_RULE,
        "segment_rule": SEGMENT_RULE,
        "metrics": METRIC_DEFINITIONS,
        "randomness": "не используется, baseline детерминирован",
        "seed": None,
        "tie_break": "по incident_id по возрастанию",
        "vectorizer": {
            "module": "src/prepare_features.py",
            "vocabulary_size": int(len(baseline.vectorizer.vocabulary_)),
            "vocabulary_sha256": vocabulary_hash(baseline.vectorizer),
            "manifest_sha256": manifest_hash,
        },
        "data_sha256": {split: sha256_file(Path(path)) for split, path in paths.items()},
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "commands": [
            "python src/generate_incidents.py --count 6000",
            "python src/prepare_dataset.py",
            "python src/split_dataset.py",
            "python src/triage_baseline.py run",
        ],
    }


def run(results_dir: Path) -> int:
    """Полный прогон: оценка двух baseline и запись артефактов."""
    paths = SPLIT_FILES
    try:
        frames = load_splits(paths)
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    train, validation = frames["train"], frames["validation"]
    vectorizer = fit_vectorizer(train)
    baseline = RetrievalBaseline(vectorizer, train, transform(vectorizer, train))

    per_query = evaluate_retrieval(baseline, validation)
    retrieval = aggregate_retrieval(per_query)
    retrieval["structure"] = label_structure(train, validation)
    classification = evaluate_classification(train, validation)
    cases = build_cases(baseline, validation, per_query, CASES_LIMIT)
    settings = build_settings(baseline, paths)

    manifest = settings["vectorizer"]["manifest_sha256"]
    if manifest is None:
        print(f"Ошибка: нет или не читается {MANIFEST_PATH}, проверка словаря невозможна")
        return 1
    if manifest != settings["vectorizer"]["vocabulary_sha256"]:
        print("Ошибка: словарь не совпадает с docs/lab6/features_manifest.json")
        return 1

    results_dir.mkdir(parents=True, exist_ok=True)
    dump = {"ensure_ascii": False, "indent": 2, "allow_nan": False}
    (results_dir / "settings.json").write_text(
        json.dumps(settings, **dump), encoding="utf-8", newline="\n"
    )
    per_query.to_csv(results_dir / "retrieval_per_query.csv", index=False, lineterminator="\n")
    (results_dir / "retrieval_metrics.json").write_text(
        json.dumps(retrieval, **dump), encoding="utf-8", newline="\n"
    )
    (results_dir / "classification_metrics.json").write_text(
        json.dumps(classification, **dump), encoding="utf-8", newline="\n"
    )
    (results_dir / "cases.json").write_text(
        json.dumps(cases, **dump), encoding="utf-8", newline="\n"
    )
    write_metrics_table(results_dir.parent / "metrics.md", retrieval, classification)
    write_cases(results_dir.parent / "cases.md", cases)

    overall = retrieval["all"]
    print(f"индекс: {len(baseline.train)} карточек train, {settings['vectorizer']['vocabulary_size']} признаков")
    print(f"словарь: {settings['vectorizer']['vocabulary_sha256'][:12]}")
    print(f"запросов: {overall['queries']} из validation, k = {TOP_K}")
    print(
        "поиск: "
        f"NDCG@3 {format_value(overall['ndcg_at_k'])}, "
        f"Recall@3 {format_value(overall['recall_at_k'])}, "
        f"MRR@3 {format_value(overall['mrr_at_k'])}, "
        f"пустых выдач {overall['empty_results']}"
    )
    structure = retrieval["structure"]
    if structure["grade_two_reachable"]:
        print(f"полная релевантность достижима: {structure['shared_service_cause_pairs']} общих пар меток")
    else:
        print("полная релевантность недостижима: общих пар «услуга + причина» нет")
    for target, report in classification.items():
        print(
            f"{target}: самый частый класс {report['majority_class']}, "
            f"F1-macro {format_value(report['f1_macro'])}, "
            f"accuracy {format_value(report['accuracy'])}"
        )
    print(f"подборка: {len(cases['success'])} успехов и {len(cases['failure'])} провалов")
    print(f"артефакты: {results_dir}")
    return 0


def search(query: str | None, query_id: str | None, k: int) -> int:
    """Интерактивная выдача по тексту или по карточке validation."""
    try:
        frames = load_splits()
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    train, validation = frames["train"], frames["validation"]
    text = (query or "").strip()
    if query_id:
        matched = validation[validation["incident_id"] == query_id]
        if matched.empty:
            print(f"Ошибка: в validation нет карточки {query_id}")
            return 1
        text = matched.iloc[0][TEXT_COLUMN]

    vectorizer = fit_vectorizer(train)
    baseline = RetrievalBaseline(vectorizer, train, transform(vectorizer, train))
    result = baseline.search(text, k)
    print(f"запрос: {text[:160]}")
    print(f"сегмент: {result.segment}, известных слов {result.known_tokens} из {result.total_tokens}")
    if not result.hits:
        print("выдача пуста: ни одно слово запроса не встречается в словаре train")
        return 0
    for rank, hit in enumerate(result.hits, start=1):
        row = baseline.labels(hit.incident_id)
        print(
            f"{rank}. {hit.incident_id} | сходство {format_value(hit.score)} | "
            f"{row['service']} / {row['root_cause']} / {row['severity']}"
        )
        print(f"   {row[TEXT_COLUMN][:140]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="оценить оба baseline и записать артефакты")
    run_parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    search_parser = subparsers.add_parser("search", help="показать выдачу по запросу")
    search_parser.add_argument("--query", help="текст запроса")
    search_parser.add_argument("--query-id", help="взять запрос из карточки validation")
    search_parser.add_argument("--k", type=int, default=TOP_K)

    args = parser.parse_args()
    if args.command == "search":
        if args.k < 1:
            parser.error("--k должно быть положительным")
        if not args.query and not args.query_id:
            parser.error("нужен --query или --query-id")
        return search(args.query, args.query_id, args.k)
    if args.command != "run":
        parser.print_help()
        return 1
    return run(Path(args.results_dir))


if __name__ == "__main__":
    sys.exit(main())
