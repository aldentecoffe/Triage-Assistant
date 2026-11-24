"""ЛР 9: контролируемый эксперимент с одним изменением — сигналом сортировки.

Гипотеза H1 и порог приёмки зафиксированы в работе 8 до этого прогона
(`docs/lab8/hypothesis.md`): переранжирование пула top-10 по вероятности той же
причины должно поднять validation NDCG@3 до 0,513 (+0,10 к baseline 0,413) без
падения защитных показателей.

Меняется ровно один фактор: порядок карточек внутри пула top-10. TF-IDF,
индекс, пул, шкала релевантности, k и набор запросов остаются прежними.
Baseline сохраняется в коде и повторяется тем же прогоном.

Запуск из корня репозитория:
    python src/experiment.py run
    python src/experiment.py freeze --decision reject --reason "..."
    python src/experiment.py test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from eval_report import check_manifest, grade_v2, ideal_counts, signal_band, write_json
from prepare_features import (
    SPLIT_FILES,
    VECTORIZER_PARAMS,
    fit_vectorizer,
    load_splits,
    transform,
    vocabulary_hash,
)
from triage_baseline import (
    RetrievalBaseline,
    format_value,
    ideal_grades,
    ndcg,
    sha256_file,
)

EXPERIMENT_VERSION = "1.0"
EXPERIMENT_ID = "E01"
POOL = 10
TOP_K = 3
# Порог перенесён из работы 8 и не пересматривается по результату прогона.
ALPHA_MARGIN = 0.10
# Прирост не должен держаться на одном сценарии: его вклад в средний ограничен.
SPREAD_LIMIT = 0.5
RESULTS_DIR = Path("docs/lab9/results")
MANIFEST_PATH = Path("docs/lab6/features_manifest.json")
LAB8_SUMMARY = Path("docs/lab8/results/summary.json")

# Классификатор причины обучается на train и нужен только как сигнал сортировки.
# Вес классов выравнен: в train capacity 42,9 %, в validation — 14,3 %, и без
# выравнивания редкая причина не выигрывает у частой.
RERANKER_NAME = "LogisticRegression"
RERANKER_PARAMS = {
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 2000,
    "class_weight": "balanced",
}

# Коды решений в машинных файлах; в документах печатаются русские названия.
DECISION_LABELS = {"include": "включить", "reject": "не включать"}
CUT_ORDER = {
    "severity": ["P1", "P2", "P3", "P4"],
    "language": ["ru", "en"],
    "signal": ["cause_closer", "signal_tie", "service_closer"],
}
CUT_TITLES = {
    "severity": "Серьёзность",
    "language": "Язык",
    "signal": "Какой сигнал ближе",
    "scenario_id": "Сценарий",
    "regime": "Режим baseline",
}
SIGNAL_TITLES = {
    "cause_closer": "причина ближе",
    "signal_tie": "сигналы равны",
    "service_closer": "услуга ближе",
}
CONDITION_TITLES = {
    "gain": "Прирост NDCG@3 не меньше 0,10",
    "p1_cause": "Причина в top-3 у P1 не ниже baseline",
    "p1_ndcg": "NDCG@3 у P1 не ниже baseline",
    "empty": "Пустых выдач ноль",
    "spread": "Прирост не держится на одном сценарии",
}


def load_selected_splits(splits: list[str]) -> dict[str, pd.DataFrame]:
    """Читает только перечисленные выборки и проверяет, что группы между ними не пересекаются.

    `prepare_features.load_splits` проверяет полноту всех трёх выборок сразу, а
    работе 9 нельзя открывать test до фиксации решения. Поэтому проверка групп
    повторяется здесь для подмножества: сценарий, событие и отпечаток текста не
    должны встречаться в двух выборках.
    """
    frames: dict[str, pd.DataFrame] = {}
    for split in splits:
        path = Path(SPLIT_FILES[split])
        try:
            frames[split] = pd.read_csv(path, dtype=str, keep_default_na=False)
        except OSError as error:
            raise FileNotFoundError(f"нет файла {path}: сначала пройдите пайплайн") from error
    combined = pd.concat(frames.values(), ignore_index=True)
    for key in ("scenario_id", "event_id", "text_hash"):
        spread = combined.groupby(key)["split"].nunique()
        crossing = spread[spread > 1]
        if len(crossing):
            examples = ", ".join(str(value) for value in crossing.index[:5])
            raise ValueError(f"{key}: {len(crossing)} групп в разных выборках ({examples})")
    return frames


def top_pool(scores: np.ndarray, ids: np.ndarray, pool: int = POOL) -> np.ndarray:
    """Индексы пула: положительное сходство, порядок по убыванию, при равенстве — по ID."""
    if pool <= 0:
        return np.empty(0, dtype=int)
    candidates = np.flatnonzero(scores > 0)
    order = np.lexsort((ids[candidates], -scores[candidates]))
    return candidates[order[:pool]]


def rerank_indices(
    pool: np.ndarray,
    scores: np.ndarray,
    ids: np.ndarray,
    probability: np.ndarray,
    k: int = TOP_K,
) -> np.ndarray:
    """Переставляет пул по вероятности причины карточки.

    Вероятности относятся к причине карточки, а не к разметке запроса: функция
    не получает метки запроса и не может их использовать. При равенстве
    вероятностей порядок задаёт косинус, затем `incident_id`.
    """
    if k <= 0 or len(pool) == 0:
        return pool[:0]
    keys = (ids[pool], -scores[pool], -probability[pool])
    order = np.lexsort(keys)
    return pool[order[:k]]


def cause_positions(causes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Номер класса для каждой карточки train: вероятность берётся по нему."""
    index = {cause: position for position, cause in enumerate(classes)}
    missing = sorted(set(causes) - set(index))
    if missing:
        raise ValueError(f"в train есть причины без обученного класса: {missing}")
    return np.array([index[cause] for cause in causes], dtype=int)


def build_reranker(train: pd.DataFrame, matrix) -> LogisticRegression:
    """Обучает классификатор причины на индексе train; других данных не видит."""
    labels = train["root_cause"].astype(str).to_numpy()
    reranker = LogisticRegression(**RERANKER_PARAMS)
    reranker.fit(matrix, labels)
    return reranker


def evaluate_frame(
    baseline: RetrievalBaseline,
    reranker: LogisticRegression,
    queries: pd.DataFrame,
) -> pd.DataFrame:
    """Считает baseline и кандидата по каждому запросу одним проходом.

    Пиксель к пикселю baseline повторяет `triage_baseline.search`: те же
    сходства, те же правила tie-break. Кандидат получает тот же пул top-10 и
    меняет только порядок.
    """
    if queries.empty:
        return empty_per_query()

    ids = baseline.train["incident_id"].to_numpy()
    services = baseline.train["service"].to_numpy()
    causes = baseline.train["root_cause"].to_numpy()
    classes = reranker.classes_
    positions = cause_positions(causes, classes)

    query_matrix = transform(baseline.vectorizer, queries)
    similarity = np.asarray((query_matrix @ baseline.matrix.T).toarray())
    pooled = reranker.predict_proba(query_matrix)
    predicted = classes[np.argmax(pooled, axis=1)]

    rows: list[dict[str, object]] = []
    for position, (_, query) in enumerate(queries.iterrows()):
        scores = similarity[position]
        probability = pooled[position][positions]
        pool = top_pool(scores, ids)
        before = pool[:TOP_K]
        after = rerank_indices(pool, scores, ids, probability)

        base_cards = [baseline.labels(ids[index]) for index in before]
        cand_cards = [baseline.labels(ids[index]) for index in after]
        grades_base = [grade_v2(query, card) for card in base_cards]
        grades_cand = [grade_v2(query, card) for card in cand_cards]

        counts = ideal_counts(baseline, query)["v2"]
        ideal = ideal_grades(*counts, TOP_K)

        same_cause = causes == query["root_cause"]
        other_cause_same_service = (services == query["service"]) & ~same_cause
        cause_max = float(scores[same_cause].max()) if same_cause.any() else 0.0
        service_max = (
            float(scores[other_cause_same_service].max())
            if other_cause_same_service.any()
            else 0.0
        )
        gap = cause_max - service_max

        rows.append(
            {
                "incident_id": query["incident_id"],
                "scenario_id": query["scenario_id"],
                "event_id": query["event_id"],
                "language": query["language"],
                "severity": query["severity"],
                "service": query["service"],
                "root_cause": query["root_cause"],
                "ideal_v2": int(counts[0] + counts[1]),
                "returned": int(len(after)),
                "returned_base": int(len(before)),
                "returned_cand": int(len(after)),
                "ids_base": " > ".join(ids[index] for index in before),
                "ids_cand": " > ".join(ids[index] for index in after),
                "grades_base": ",".join(str(grade) for grade in grades_base),
                "grades_cand": ",".join(str(grade) for grade in grades_cand),
                "regime": ",".join(str(grade) for grade in grades_base),
                "ndcg_base": round(ndcg(grades_base, ideal, TOP_K), 6),
                "ndcg_cand": round(ndcg(grades_cand, ideal, TOP_K), 6),
                "recall_base": int(any(grade >= 1 for grade in grades_base)),
                "recall_cand": int(any(grade >= 1 for grade in grades_cand)),
                "cause_hit_base": int(any(grade == 2 for grade in grades_base)),
                "cause_hit_cand": int(any(grade == 2 for grade in grades_cand)),
                "service_hit_base": int(
                    any(card["service"] == query["service"] for card in base_cards)
                ),
                "service_hit_cand": int(
                    any(card["service"] == query["service"] for card in cand_cards)
                ),
                "classifier_top1": int(predicted[position] == query["root_cause"]),
                "predicted_cause": str(predicted[position]),
                "gap": round(gap, 6),
                "signal": signal_band(gap),
                "changed": int(list(before) != list(after)),
            }
        )

    frame = pd.DataFrame(rows)
    frame["delta"] = (frame["ndcg_cand"] - frame["ndcg_base"]).round(6)
    return frame


def empty_per_query() -> pd.DataFrame:
    """Пустая таблица с теми же колонками: сводка на пустом входе не падает."""
    columns = {
        "incident_id": "string",
        "scenario_id": "string",
        "event_id": "string",
        "language": "string",
        "severity": "string",
        "service": "string",
        "root_cause": "string",
        "ideal_v2": "int64",
        "returned": "int64",
        "returned_base": "int64",
        "returned_cand": "int64",
        "ids_base": "string",
        "ids_cand": "string",
        "grades_base": "string",
        "grades_cand": "string",
        "regime": "string",
        "ndcg_base": "float64",
        "ndcg_cand": "float64",
        "recall_base": "int64",
        "recall_cand": "int64",
        "cause_hit_base": "int64",
        "cause_hit_cand": "int64",
        "service_hit_base": "int64",
        "service_hit_cand": "int64",
        "classifier_top1": "int64",
        "predicted_cause": "string",
        "gap": "float64",
        "signal": "string",
        "changed": "int64",
        "delta": "float64",
    }
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns.items()})


def usable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Запросы, у которых в индексе есть хоть одна релевантная карточка."""
    if frame.empty:
        return frame
    return frame[frame["ideal_v2"] > 0]


def _mean(frame: pd.DataFrame, column: str) -> float:
    """Среднее по колонке; на пустом наборе — ноль, а не NaN."""
    return round(float(frame[column].mean()), 6) if len(frame) else 0.0


def mean_ndcg(
    baseline: RetrievalBaseline,
    queries: pd.DataFrame,
    similarity: np.ndarray | None = None,
    candidate_scores: np.ndarray | None = None,
) -> float:
    """Средний NDCG@3 при готовых оценках карточек; без оценок — baseline.

    Нужен для сравнения сигналов сортировки: baseline даёт `candidate_scores=None`,
    любой вариант — матрицу «запрос × карточка». Порядок пула и tie-break те же,
    что в `evaluate_frame`.
    """
    if queries.empty:
        return 0.0
    ids = baseline.train["incident_id"].to_numpy()
    if similarity is None:
        query_matrix = transform(baseline.vectorizer, queries)
        similarity = np.asarray((query_matrix @ baseline.matrix.T).toarray())
    values = []
    for position, (_, query) in enumerate(queries.iterrows()):
        scores = similarity[position]
        pool = top_pool(scores, ids)
        after = (
            pool[:TOP_K]
            if candidate_scores is None
            else rerank_indices(pool, scores, ids, candidate_scores[position])
        )
        grades = [grade_v2(query, baseline.labels(ids[index])) for index in after]
        counts = ideal_counts(baseline, query)["v2"]
        # То же правило, что в summarise: запрос без релевантных карточек в индексе
        # не входит в среднее, иначе baseline в variants разойдётся со summary.
        if counts[0] + counts[1] == 0:
            continue
        values.append(ndcg(grades, ideal_grades(*counts, TOP_K), TOP_K))
    return round(float(np.mean(values)), 6) if values else 0.0


def metrics_block(frame: pd.DataFrame, suffix: str) -> dict[str, float]:
    """Набор метрик одного варианта сортировки."""
    return {
        "ndcg@3": _mean(frame, f"ndcg_{suffix}"),
        "recall@3": _mean(frame, f"recall_{suffix}"),
        "cause_hit@3": _mean(frame, f"cause_hit_{suffix}"),
        "service_hit@3": _mean(frame, f"service_hit_{suffix}"),
    }


def summarise(frame: pd.DataFrame) -> dict[str, object]:
    """Сводка по validation: главная метрика, защитные, диагностика переранжировщика."""
    scored = usable_frame(frame)
    baseline = metrics_block(scored, "base")
    candidate = metrics_block(scored, "cand")
    delta = {
        key: round(candidate[key] - baseline[key], 6) for key in baseline
    }
    p1 = scored[scored["severity"] == "P1"] if len(scored) else scored
    by_cause = []
    total_queries = len(scored)
    for cause, group in scored.groupby("root_cause", sort=True):
        base = _mean(group, "ndcg_base")
        cand = _mean(group, "ndcg_cand")
        by_cause.append(
            {
                "cause": cause,
                "queries": int(len(group)),
                "accuracy": round(float(group["classifier_top1"].mean()), 6),
                "ndcg_base": base,
                "ndcg_cand": cand,
                "delta": round(cand - base, 6),
                "contribution": round((cand - base) * len(group) / total_queries, 6)
                if total_queries
                else 0.0,
            }
        )
    predicted_counts = (
        {str(k): int(v) for k, v in scored["predicted_cause"].value_counts().items()}
        if len(scored)
        else {}
    )
    saturated = max(predicted_counts, key=predicted_counts.get) if predicted_counts else ""
    saturated_queries = predicted_counts.get(saturated, 0)
    overall_delta = float(delta["ndcg@3"])
    saturated_contribution = next(
        (row["contribution"] for row in by_cause if row["cause"] == saturated), 0.0
    )
    saturated_gain_share = (
        round(float(saturated_contribution) / overall_delta, 6) if overall_delta > 0 else None
    )
    return {
        "queries": int(len(frame)),
        "scored_queries": int(len(scored)),
        "excluded": int(len(frame) - len(scored)),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "protective": {
            "rule": "причина в top-3 у P1 и NDCG@3 у P1 не ниже baseline; пустых выдач 0",
            "p1_queries": int(len(p1)),
            "p1_cause_hit_base": _mean(p1, "cause_hit_base"),
            "p1_cause_hit_cand": _mean(p1, "cause_hit_cand"),
            "p1_ndcg_base": _mean(p1, "ndcg_base"),
            "p1_ndcg_cand": _mean(p1, "ndcg_cand"),
            "empty_base": int((frame["returned_base"] == 0).sum()) if len(frame) else 0,
            "empty_cand": int((frame["returned_cand"] == 0).sum()) if len(frame) else 0,
        },
        "reranker": {
            "params": {"model": RERANKER_NAME, **RERANKER_PARAMS},
            "accuracy": _mean(scored, "classifier_top1"),
            "by_cause": by_cause,
            "predicted": predicted_counts,
            "saturated_cause": saturated,
            "saturated_queries": int(saturated_queries),
            "saturated_share": round(saturated_queries / len(scored), 6) if len(scored) else 0.0,
            "saturated_gain_share": saturated_gain_share,
        },
        "changed_queries": int(scored["changed"].sum()) if len(scored) else 0,
        "improved_queries": int((scored["delta"] > 0).sum()) if len(scored) else 0,
        "worsened_queries": int((scored["delta"] < 0).sum()) if len(scored) else 0,
    }


def decision_checks(frame: pd.DataFrame, summary: dict[str, object]) -> dict[str, object]:
    """Заранее заданные условия H1, посчитанные по факту прогона.

    Порог прироста и защитные условия берутся из карточки гипотезы работы 8.
    Условие распределённости добавлено там же: прирост, который держится на
    одном сценарии, не считается подтверждением.
    """
    scored = usable_frame(frame)
    delta = float(summary["delta"]["ndcg@3"])  # type: ignore[index]
    protective = summary["protective"]  # type: ignore[assignment]
    contributions: dict[str, float] = {}
    if len(scored) and delta > 0:
        grouped = scored.groupby("scenario_id")
        for scenario, group in grouped:
            contributions[str(scenario)] = round(
                float(group["delta"].mean()) * len(group) / len(scored), 6
            )
    dominant = max(contributions, key=contributions.get) if contributions else ""
    dominant_contribution = contributions.get(dominant, 0.0)
    share = round(dominant_contribution / delta, 6) if delta > 0 else 1.0
    checks = {
        "gain": round(delta, 6) >= ALPHA_MARGIN,
        # Без P1-запросов защиту подтвердить нечем: 0 >= 0 было бы выполнением на пустом месте.
        "p1_cause": int(protective["p1_queries"]) > 0
        and float(protective["p1_cause_hit_cand"]) >= float(protective["p1_cause_hit_base"]),
        "p1_ndcg": int(protective["p1_queries"]) > 0
        and float(protective["p1_ndcg_cand"]) >= float(protective["p1_ndcg_base"]),
        "empty": int(protective["empty_cand"]) == 0,
        "spread": bool(contributions) and share <= SPREAD_LIMIT,
    }
    return {
        "rules": {
            "alpha_margin": ALPHA_MARGIN,
            "alpha_rule": "baseline NDCG@3 + 0,10",
            "alpha": round(float(summary["baseline"]["ndcg@3"]) + ALPHA_MARGIN, 6),  # type: ignore[index]
            "protective": protective,
            "spread_limit": SPREAD_LIMIT,
        },
        "checks": checks,
        "delta": delta,
        "dominant_scenario": dominant,
        "dominant_contribution": dominant_contribution,
        "dominant_share": share,
        "scenario_contributions": contributions,
    }


def expected_decision(checks: dict[str, bool]) -> str:
    """Решение по заранее заданным условиям: все выполнены — включать.

    Пустой набор условий не считается выполненным: отсутствие проверок не даёт
    права включать изменение, и `freeze` такую сводку тоже не примет.
    """
    return "include" if checks and all(checks.values()) else "reject"


def config_payload() -> dict[str, object]:
    """Конфигурация запуска: один изменяемый фактор и всё, что зафиксировано."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": EXPERIMENT_VERSION,
        "baseline": {"name": "tfidf_cosine_top3", "pool": None, "reranker": None},
        "candidate": {
            "name": "tfidf_cosine_pool10_cause_rerank",
            "pool": POOL,
            "reranker": {"model": RERANKER_NAME, **RERANKER_PARAMS},
        },
        "k": TOP_K,
        "main_metric": "ndcg@3 по шкале v2.0",
        "grading_rule": "2 — та же причина; 1 — та же услуга при другой причине; 0 — иначе",
        "selection_split": "validation",
        "final_split": "test",
        "data_version": "1.1",
        "vectorizer": VECTORIZER_PARAMS,
        "seed": None,
        "alpha_rule": f"baseline NDCG@3 + 0,{int(ALPHA_MARGIN * 100):02d}",
        "spread_limit": SPREAD_LIMIT,
    }


def config_sha256(config: dict[str, object]) -> str:
    """Отпечаток конфигурации: видно, что между запусками она не менялась."""
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def environment() -> dict[str, str]:
    """Версии, от которых зависит результат прогона."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def input_hashes(splits: list[str]) -> dict[str, str]:
    """Отпечатки файлов данных, которые участвовали в прогоне."""
    return {
        split: sha256_file(Path(SPLIT_FILES[split]))
        for split in splits
        if Path(SPLIT_FILES[split]).exists()
    }


def reproduction_check(summary: dict[str, object]) -> dict[str, object]:
    """Сверка baseline с опубликованной работой 8: метрика должна повториться."""
    block: dict[str, object] = {"reference": str(LAB8_SUMMARY), "checked": False}
    if not LAB8_SUMMARY.exists():
        block["note"] = "сводки работы 8 нет: сверка пропущена"
        return block
    reference = json.loads(LAB8_SUMMARY.read_text(encoding="utf-8"))
    expected = float(reference["ndcg"]["v2"])
    actual = float(summary["baseline"]["ndcg@3"])  # type: ignore[index]
    block.update(
        {
            "checked": True,
            "ndcg_lab8": round(expected, 6),
            "ndcg_now": round(actual, 6),
            "difference": round(actual - expected, 6),
            "match": abs(actual - expected) < 1e-6,
            "ceiling": reference.get("ceiling", {}),
        }
    )
    return block


def build_summary(
    frame: pd.DataFrame, train_causes: list[str] | None = None
) -> dict[str, object]:
    """Сводка вместе с конфигурацией, отпечатками и условиями решения."""
    summary = summarise(frame)
    config = config_payload()
    summary["config"] = config
    summary["config_sha256"] = config_sha256(config)
    summary["hashes"] = input_hashes(["train", "validation"])
    summary["vocabulary_sha256"] = ""
    summary["environment"] = environment()
    summary["reproduction"] = reproduction_check(summary)
    summary["decision_checks"] = decision_checks(frame, summary)
    summary["uncovered"] = uncovered_queries(frame, train_causes)
    summary["train_cause_share"] = {}
    return summary


def uncovered_queries(
    frame: pd.DataFrame, train_causes: list[str] | None = None
) -> dict[str, object]:
    """Что проверка не покрывает: причины, сценарии, языки, пустые выдачи.

    Список причин берётся из train, а не из константы: если состав таксономии
    изменится, документ назовёт ровно те причины, которых нет в валидации.
    """
    train_causes = sorted(train_causes or [])
    if frame.empty:
        return {
            "causes": [],
            "missing_causes": train_causes,
            "scenarios": 0,
            "scenarios_list": [],
            "p1_scenarios": [],
            "languages": {},
            "no_relevant": 0,
            "total_causes_in_taxonomy": len(train_causes),
            "train_causes_known": bool(train_causes),
        }
    causes = sorted(frame["root_cause"].unique())
    scenarios = sorted(frame["scenario_id"].unique())
    p1 = frame[frame["severity"] == "P1"]
    missing = [cause for cause in train_causes if cause not in causes] if train_causes else []
    return {
        "causes": causes,
        "missing_causes": missing,
        "scenarios": len(scenarios),
        "scenarios_list": scenarios,
        "p1_scenarios": sorted(p1["scenario_id"].unique()) if len(p1) else [],
        "languages": {str(k): int(v) for k, v in frame["language"].value_counts().items()},
        "no_relevant": int((frame["ideal_v2"] == 0).sum()),
        "total_causes_in_taxonomy": len(train_causes) if train_causes else len(causes),
        "train_causes_known": bool(train_causes),
    }


def segment_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Разрезы: серьёзность, язык, сигнал, сценарий, режим baseline."""
    scored = usable_frame(frame)
    cuts = ["severity", "language", "signal", "scenario_id", "regime"]
    rows: list[dict[str, object]] = [
        {
            "cut": "Validation",
            "value": "все запросы",
            "queries": int(len(scored)),
            "ndcg_base": _mean(scored, "ndcg_base"),
            "ndcg_cand": _mean(scored, "ndcg_cand"),
            "delta": round(_mean(scored, "ndcg_cand") - _mean(scored, "ndcg_base"), 6),
            "improved": int((scored["delta"] > 0).sum()) if len(scored) else 0,
            "worsened": int((scored["delta"] < 0).sum()) if len(scored) else 0,
        }
    ]
    for cut in cuts:
        if scored.empty:
            continue
        order = CUT_ORDER.get(cut, [])
        values = [value for value in order if value in set(scored[cut])]
        values += sorted(set(scored[cut]) - set(values))
        buckets: dict[str, list[pd.DataFrame]] = {}
        for value in values:
            label = value
            if cut == "regime":
                if value == "":
                    label = "пустая выдача"
                elif value not in ("2,2,2", "1,1,1", "0,0,0"):
                    label = "смешанный"
            buckets.setdefault(label, []).append(scored[scored[cut] == value])
        labels = list(buckets)
        if cut == "regime":
            labels = [
                label
                for label in ("2,2,2", "1,1,1", "0,0,0", "смешанный", "пустая выдача")
                if label in buckets
            ]
        for label in labels:
            group = pd.concat(buckets[label])
            base = _mean(group, "ndcg_base")
            cand = _mean(group, "ndcg_cand")
            rows.append(
                {
                    "cut": CUT_TITLES[cut],
                    "value": SIGNAL_TITLES.get(label, str(label)),
                    "queries": int(len(group)),
                    "ndcg_base": base,
                    "ndcg_cand": cand,
                    "delta": round(cand - base, 6),
                    "improved": int((group["delta"] > 0).sum()),
                    "worsened": int((group["delta"] < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def pick_examples(frame: pd.DataFrame) -> dict[str, object]:
    """Один изменившийся и один неизменившийся пример выдачи."""
    scored = usable_frame(frame)
    result: dict[str, object] = {"changed": None, "unchanged": None}
    if scored.empty:
        return result
    changed = scored[scored["changed"] == 1].sort_values(
        ["delta", "incident_id"], ascending=[False, True]
    )
    if len(changed):
        result["changed"] = changed.iloc[0].to_dict()
    unchanged = scored[(scored["changed"] == 0) & (scored["delta"] == 0)]
    if len(unchanged):
        preferred = unchanged[unchanged["regime"] != "2,2,2"]
        result["unchanged"] = (preferred if len(preferred) else unchanged).iloc[0].to_dict()
    return result


def format_cards(row: dict[str, object], key: str) -> str:
    """Карточки с оценками релевантности: `I000123 (2) > ...`."""
    raw_ids = row.get(f"ids_{key}")
    raw_grades = row.get(f"grades_{key}")
    ids = str(raw_ids).split(" > ") if isinstance(raw_ids, str) and raw_ids else []
    grades = str(raw_grades).split(",") if isinstance(raw_grades, str) and raw_grades else []
    if not ids:
        return "пустая выдача"
    return " > ".join(
        f"{incident} ({grade})" for incident, grade in zip(ids, grades)
    )


def comparison_block(summary: dict[str, object], segments: pd.DataFrame) -> str:
    """Общая таблица: baseline, кандидат и прирост по сегментам."""
    lines = [
        "| Выборка и сегмент | Запросов | Baseline NDCG@3 | Кандидат NDCG@3 | Прирост | Улучшилось / ухудшилось |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in segments.iterrows():
        lines.append(
            "| {cut}: {value} | {queries} | {base} | {cand} | {delta} | {improved} / {worsened} |".format(
                cut=row["cut"],
                value=row["value"],
                queries=format_int(row["queries"]),
                base=format_value(float(row["ndcg_base"])),
                cand=format_value(float(row["ndcg_cand"])),
                delta=format_value(float(row["delta"])),
                improved=row["improved"],
                worsened=row["worsened"],
            )
        )
    return "\n".join(lines)


def examples_block(examples: dict[str, object]) -> str:
    """Два примера: один изменился, другой нет."""
    lines = [
        "| Запрос | Top-3 baseline | Top-3 кандидата | NDCG baseline | NDCG кандидата |",
        "|---|---|---|---|---|",
    ]
    for key, title in (("changed", "Изменился"), ("unchanged", "Не изменился")):
        row = examples.get(key)
        if row is None:
            lines.append(f"| {title}: нет подходящего примера | — | — | — | — |")
            continue
        lines.append(
            "| {title} {incident} ({cause}, {service}) | {base} | {cand} | {nb} | {nc} |".format(
                title=title,
                incident=row["incident_id"],
                cause=row["root_cause"],
                service=row["service"],
                base=format_cards(row, "base"),
                cand=format_cards(row, "cand"),
                nb=format_value(float(row["ndcg_base"])),
                nc=format_value(float(row["ndcg_cand"])),
            )
        )
    return "\n".join(lines)


def protective_block(summary: dict[str, object]) -> str:
    """Защитные показатели: пороги из работы 8 и фактическое значение.

    Слово «обнулилось» здесь выбирается по числам, а не вписано: генератор
    должен оставаться правдивым и на другом прогоне.
    """
    protective = summary["protective"]  # type: ignore[assignment]
    queries = int(protective["p1_queries"])
    base_count = round(float(protective["p1_cause_hit_base"]) * queries)
    cand_count = round(float(protective["p1_cause_hit_cand"]) * queries)
    if cand_count == 0 and base_count > 0:
        verb = "обнулилось"
    elif cand_count > base_count:
        verb = "выросло"
    elif cand_count < base_count:
        verb = "снизилось"
    else:
        verb = "не изменилось"
    p1_delta = float(protective["p1_ndcg_cand"]) - float(protective["p1_ndcg_base"])
    lines = [
        "| Показатель | Baseline | Кандидат | Условие |",
        "|---|---|---|---|",
        "| Причина в top-3 у P1 | {base} | {cand} | не ниже baseline |".format(
            base=format_value(float(protective["p1_cause_hit_base"])),
            cand=format_value(float(protective["p1_cause_hit_cand"])),
        ),
        "| NDCG@3 у P1 | {base} | {cand} | не ниже baseline |".format(
            base=format_value(float(protective["p1_ndcg_base"]), 4),
            cand=format_value(float(protective["p1_ndcg_cand"]), 4),
        ),
        "| Пустых выдач | {base} | {cand} | 0 |".format(
            base=protective["empty_base"], cand=protective["empty_cand"]
        ),
        "",
    ]
    if queries == 0:
        lines.append("В validation нет запросов P1: защитные условия не проверялись.")
        return "\n".join(lines)
    lines.append(
        "Попадание причины в top-3 у P1 {verb}: из {queries} запросов P1 "
        "baseline находил причину у {base}, кандидат — у {cand}.".format(
            verb=verb, queries=queries, base=base_count, cand=cand_count
        )
    )
    note = "Прирост NDCG@3 у P1 — {delta}.".format(delta=format_value(p1_delta, 4))
    if abs(p1_delta) < 0.0005:
        note += (
            " В сегментной таблице с тремя знаками он округляется до нуля, поэтому "
            "защитный блок печатает четыре."
        )
    lines.append(note)
    return "\n".join(lines)


def reranker_block(summary: dict[str, object]) -> str:
    """Точность классификатора причины по классам: проверка риска из ЛР 8."""
    reranker = summary["reranker"]  # type: ignore[assignment]
    by_cause = reranker["by_cause"]  # type: ignore[assignment]
    lines = [
        f"Точность top-1 на validation: {format_value(float(reranker['accuracy']))} "
        f"на {format_int(summary['scored_queries'])} запросах. Классификатор обучался только на train; "
        "его ответ не используется в оценке релевантности, только как сигнал сортировки.",
        "",
        "| Причина запроса | Запросов | Точность top-1 |",
        "|---|---|---|",
    ]
    for row in by_cause:  # type: ignore[union-attr]
        lines.append(
            "| {cause} | {queries} | {accuracy} |".format(
                cause=row["cause"],
                queries=row["queries"],
                accuracy=format_value(float(row["accuracy"])),
            )
        )
    lines += ["", saturation_sentence(summary)]
    return "\n".join(lines)


def uncovered_block(summary: dict[str, object]) -> str:
    """Классы запросов, которые проверкой не покрыты."""
    uncovered = summary["uncovered"]  # type: ignore[assignment]
    languages = ", ".join(
        f"{name}: {count}" for name, count in uncovered["languages"].items()  # type: ignore[union-attr]
    )
    causes = ", ".join(uncovered["causes"]) if uncovered["causes"] else "нет"  # type: ignore[index]
    missing = uncovered.get("missing_causes") or []  # type: ignore[union-attr]
    known = bool(uncovered.get("train_causes_known"))
    if missing:
        if len(missing) == 1:
            cause_line = (
                f"- Причины в validation: {causes}. Вне проверки осталась причина "
                f"{missing[0]}: она есть в train, но не в validation."
            )
        else:
            cause_line = (
                f"- Причины в validation: {causes}. Вне проверки остались причины "
                f"{', '.join(missing)}: они есть в train, но не в validation."
            )
    elif known:
        cause_line = f"- Причины в validation: {causes}; все причины train покрыты."
    else:
        cause_line = (
            f"- Причины в validation: {causes}; список причин train в сводке не записан, "
            "полнота покрытия не проверялась."
        )
    p1_scenarios = uncovered["p1_scenarios"]  # type: ignore[index]
    if p1_scenarios:
        p1_line = (
            f"- Сценариев в validation: {uncovered['scenarios']}; P1 — только "  # type: ignore[index]
            f"{', '.join(p1_scenarios)}. Защитный показатель P1 измеряет знакомство "
            "с лексикой одного сценария."
        )
    else:
        p1_line = (
            f"- Сценариев в validation: {uncovered['scenarios']}; запросов P1 нет, "  # type: ignore[index]
            "поэтому защитные условия P1 не проверялись."
        )
    lines = [
        cause_line,
        p1_line,
        f"- Языки: {languages or 'не указаны'}. Английские запросы есть, но их меньше, "
        "и отдельного порога для них не задано.",
        f"- Запросов без релевантных карточек в индексе: {uncovered['no_relevant']}. "  # type: ignore[index]
        "Правило исключения из среднего записано в работе 8 и здесь не пересматривается.",
        "- Длина описания и редкие интерфейсы отдельными сегментами не проверялись: "
        "их разбор остался в работе 8.",
    ]
    return "\n".join(lines)


def render_comparison(
    summary: dict[str, object],
    frame: pd.DataFrame,
    test_summary: dict[str, object] | None,
) -> str:
    """Сравнительная таблица: validation целиком, сегменты, примеры, test."""
    if not summary.get("uncovered"):
        summary = {**summary, "uncovered": uncovered_queries(frame)}
    segments = segment_table(frame)
    examples = pick_examples(frame)
    lines = [
        "# Сравнительная таблица",
        "",
        f"Эксперимент {EXPERIMENT_ID}. Единица — запрос validation, таких запросов "
        f"{format_int(summary['scored_queries'])}; выдача — top-3 похожих карточек из train. "
        "Baseline и кандидат считаются одним прогоном на одних и тех же данных, "
        "индексе и пуле.",
        "",
        "## Validation: общий результат и сегменты",
        "",
        comparison_block(summary, segments),
        "",
        "## Изменившийся и неизменившийся пример",
        "",
        examples_block(examples),
        "",
        "## Защитные показатели",
        "",
        protective_block(summary),
        "",
        "## Переранжировщик: проверка риска «частая причина»",
        "",
        reranker_block(summary),
        "",
        "## Что проверка не покрывает",
        "",
        uncovered_block(summary),
        "",
    ]
    if test_summary is None:
        lines += [
            "## Test",
            "",
            "Test не запускался: до фиксации решения он не читается. Команда "
            "`python src/experiment.py test` работает только после `freeze`.",
        ]
    else:
        lines += [
            "## Test: контроль после фиксации решения",
            "",
            "| Выборка | Запросов | Baseline NDCG@3 | Кандидат NDCG@3 | Прирост |",
            "|---|---|---|---|---|",
            "| Test, все | {queries} | {base} | {cand} | {delta} |".format(
                queries=format_int(test_summary["scored_queries"]),
                base=format_value(float(test_summary["baseline"]["ndcg@3"])),  # type: ignore[index]
                cand=format_value(float(test_summary["candidate"]["ndcg@3"])),  # type: ignore[index]
                delta=format_value(float(test_summary["delta"]["ndcg@3"])),  # type: ignore[index]
            ),
            "",
            "Test не использовался для выбора: решение зафиксировано до этого прогона, "
            "и строка показывает только контроль. Вторым независимым исследованием она "
            "не является: test сгенерирован тем же генератором и делит с train часть "
            "словаря, хотя ни один шаблон описаний не встречается в двух выборках "
            "(проверка работы 5).",
        ]
    return "\n".join(lines) + "\n"


def render_log(
    summary: dict[str, object],
    decision: dict[str, object] | None,
    test_summary: dict[str, object] | None,
) -> str:
    """Журнал запуска: дата, автор, конфигурация, отпечатки, пути, решение."""
    hashes = summary["hashes"]  # type: ignore[assignment]
    reproduction = summary["reproduction"]  # type: ignore[assignment]
    environment_block = summary["environment"]  # type: ignore[assignment]
    lines = [
        "# Журнал эксперимента",
        "",
        "## Идентификация запуска",
        "",
        "| Поле | Значение |",
        "|---|---|",
        f"| Код эксперимента | {EXPERIMENT_ID} |",
        f"| Версия модуля | {EXPERIMENT_VERSION} |",
        f"| Дата прогона | {summary.get('run_date', 'не указана')} |",
        "| Автор прогона | Плотников М. А., учебный проект; прогон выполнен модулем "
        "`src/experiment.py` |",
        "| Изменение | сигнал сортировки: пул top-10 переранжируется по вероятности "
        "причины карточки |",
        "| Команда | `python src/experiment.py run` |",
        "| Код изменения | `src/experiment.py`, функции `build_reranker`, "
        "`rerank_indices`, `evaluate_frame` |",
        f"| Отбор | {summary['config']['selection_split']}, {format_int(summary['queries'])} запросов |",  # type: ignore[index]
        "| Контроль | test после фиксации решения, отдельная команда |",
        "",
        "## Конфигурация запуска",
        "",
        "```json",
        json.dumps(summary["config"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Контроль условий",
        "",
        "| Условие | Значение |",
        "|---|---|",
        f"| Версия данных | {summary['config']['data_version']} |",  # type: ignore[index]
        f"| Словарь TF-IDF | `{summary['vocabulary_sha256']}` |",
        f"| Отпечаток конфигурации | `{summary['config_sha256']}` |",
        f"| Train, SHA-256 | `{hashes.get('train', 'не записан')}` |",  # type: ignore[union-attr]
        f"| Validation, SHA-256 | `{hashes.get('validation', 'не записан')}` |",  # type: ignore[union-attr]
        f"| Python | {environment_block['python']} |",  # type: ignore[index]
        f"| scikit-learn | {environment_block['scikit_learn']} |",  # type: ignore[index]
        f"| pandas | {environment_block['pandas']} |",  # type: ignore[index]
        f"| numpy | {environment_block['numpy']} |",  # type: ignore[index]
        f"| Платформа | {environment_block['platform']} |",  # type: ignore[index]
        "",
        "Случайность не используется: seed не задан, повторный запуск даёт те же "
        "оценки, поэтому десять повторов не добавили бы независимых наблюдений.",
        "",
        "## Результаты",
        "",
        "| Файл | Что внутри |",
        "|---|---|",
        "| `docs/lab9/results/summary.json` | сводка, условия решения, отпечатки, окружение |",
        "| `docs/lab9/results/per_query.csv` | метрики каждого запроса до и после |",
        "| `docs/lab9/results/segments.csv` | разрезы из сравнительной таблицы |",
        "| `docs/lab9/results/config.json` | конфигурация запуска |",
        "| `docs/lab9/results/decision.json` | зафиксированное решение и проверки |",
        "| `docs/lab9/results/test_summary.json` | контроль на test после решения |",
        "| `docs/lab9/results/variants.json` | сравнение четырёх сигналов на validation |",
        "",
        "Сверка baseline с работой 8: {verdict}.".format(
            verdict=(
                f"NDCG@3 {format_value(float(reproduction['ndcg_now']))} против "
                f"{format_value(float(reproduction['ndcg_lab8']))}, разница "
                f"{format_value(float(reproduction['difference']))}, совпадает"
                if reproduction.get("match")
                else "сводки для сверки нет или числа расходятся"
            )
        ),
        "",
        "## Решение",
        "",
    ]
    if decision is None:
        lines.append(
            "Решение не зафиксировано. До `python src/experiment.py freeze` test не "
            "читается."
        )
    else:
        lines.append(
            f"Решение E01: **{DECISION_LABELS[str(decision['decision'])]}**. "
            f"Причина: {decision['reason']}. Проверки: "
            + ", ".join(
                f"{name} — {'выполнено' if value else 'не выполнено'}"
                for name, value in decision["checks"].items()  # type: ignore[union-attr]
            )
            + ". Расшифровка условий — в `decision.md`."
        )
    if test_summary is not None:
        lines += [
            "",
            f"Контроль на test: baseline {format_value(float(test_summary['baseline']['ndcg@3']))}, "  # type: ignore[index]
            f"кандидат {format_value(float(test_summary['candidate']['ndcg@3']))}, "  # type: ignore[index]
            f"прирост {format_value(float(test_summary['delta']['ndcg@3']))}. "  # type: ignore[index]
            "Test не влиял на выбор.",
        ]
    return "\n".join(lines) + "\n"


def capitalise(text: str) -> str:
    """Причина решения начинается с заглавной буквы, где бы её ни напечатали."""
    return text[:1].upper() + text[1:] if text else text


def format_int(value: object) -> str:
    """Число с пробелом между тысячами: «1 200» вместо «1200»."""
    return f"{int(value):,}".replace(",", " ")


def format_share(value: float) -> str:
    """Доля в процентах: 0,261 читается как «26 %»."""
    return f"{value * 100:.0f} %"


def direction(value: float, up: str = "выросло", down: str = "упало") -> str:
    """Глагол по знаку изменения: вывод в документе не должен быть константой."""
    if value > 0:
        return up
    if value < 0:
        return down
    return "не изменилось"


def signal_sentence(summary: dict[str, object]) -> str:
    """Что сделал сигнал с попаданиями причины и услуги — по фактическим числам."""
    base_cause = float(summary["baseline"]["cause_hit@3"])  # type: ignore[index]
    cand_cause = float(summary["candidate"]["cause_hit@3"])  # type: ignore[index]
    base_service = float(summary["baseline"]["service_hit@3"])  # type: ignore[index]
    cand_service = float(summary["candidate"]["service_hit@3"])  # type: ignore[index]
    cause_delta = cand_cause - base_cause
    service_delta = cand_service - base_service
    text = (
        "Сигнал сортировки изменил выдачу так: попадание причины в top-3 {verb_cause} "
        "с {base_cause} до {cand_cause}, попадание услуги {verb_service} с {base_service} "
        "до {cand_service}."
    ).format(
        verb_cause=direction(cause_delta),
        base_cause=format_value(base_cause),
        cand_cause=format_value(cand_cause),
        verb_service=direction(service_delta),
        base_service=format_value(base_service),
        cand_service=format_value(cand_service),
    )
    if cause_delta > 0 and service_delta < 0:
        text += " Рост средней метрики оплачен услугами, которые дежурный видел раньше."
    elif cause_delta > 0:
        text += " Услуга при этом не пострадала."
    else:
        text += " Ожидаемого сдвига в сторону причины нет."
    return text


def p1_boundary(summary: dict[str, object]) -> str:
    """Оговорка про P1 строится по сценариям из сводки, а не вписана руками."""
    uncovered = summary.get("uncovered") if isinstance(summary.get("uncovered"), dict) else {}
    scenarios = uncovered.get("p1_scenarios") or []
    if len(scenarios) == 1:
        return (
            f"P1 — один сценарий {scenarios[0]}, защитный показатель P1 измеряет "
            "знакомство с его лексикой."
        )
    if scenarios:
        return (
            f"P1 охватывает сценарии {', '.join(scenarios)}; защитный показатель P1 "
            "читается как среднее по ним."
        )
    return "В validation нет запросов P1: защитные условия P1 не проверялись."


def saturation_sentence(summary: dict[str, object]) -> str:
    """Насколько сигнал сведён к одной причине и откуда взялся прирост.

    Насыщение и вклад в прирост — разные вещи: классификатор может почти всегда
    отвечать одной причиной, а прирост приходить из запросов других причин.
    Поэтому вывод выбирается по доле насыщенного класса в приросте, а не по доле
    его argmax.
    """
    reranker = summary.get("reranker")
    reranker = reranker if isinstance(reranker, dict) else {}
    share = float(reranker.get("saturated_share") or 0.0)
    if share < 0.5:
        return "Argmax классификатора распределён по причинам, насыщения одним классом нет."
    cause = str(reranker.get("saturated_cause") or "")
    by_cause = reranker.get("by_cause") or []
    worst = min(by_cause, key=lambda row: float(row["accuracy"])) if by_cause else None
    text = (
        "Сигнал насыщен классом `{cause}`: argmax — {cause} у {queries} из {total} "
        "запросов ({share}).".format(
            cause=cause,
            queries=int(reranker.get("saturated_queries") or 0),
            total=format_int(summary["scored_queries"]),
            share=format_share(share),
        )
    )
    if worst is not None:
        if float(worst["accuracy"]) == 0.0:
            text += f" Причина `{worst['cause']}` не распознана ни разу."
        else:
            text += (
                f" Минимальная точность по причине — {worst['cause']} "
                f"{format_value(float(worst['accuracy']))}."
            )
    total_delta = float(summary["delta"]["ndcg@3"])  # type: ignore[index]
    if total_delta <= 0:
        text += " Прироста нет, поэтому вклады причин не считаются."
        return text
    gain_share = float(reranker.get("saturated_gain_share") or 0.0)
    if gain_share >= 0.5:
        text += " Прирост в основном переставляет карточки внутри этого класса."
    else:
        ordered = sorted(
            by_cause, key=lambda row: -float(row.get("contribution") or 0.0)
        )
        pieces = ", ".join(
            f"{row['cause']} — {format_value(float(row.get('contribution') or 0.0))}"
            for row in ordered
        )
        text += (
            " Прирост {delta} при этом распределён по причинам: {pieces}. "
            "Насыщение ограничивает интерпретацию сигнала, но не объясняет весь эффект."
        ).format(delta=format_value(float(summary["delta"]["ndcg@3"])), pieces=pieces)  # type: ignore[index]
    return text


def render_decision(
    summary: dict[str, object],
    decision: dict[str, object],
    test_summary: dict[str, object] | None = None,
) -> str:
    """Интерпретация и решение: условия, границы вывода, следующий шаг."""
    checks = decision["checks"]  # type: ignore[assignment]
    checks_block = "\n".join(
        f"| {CONDITION_TITLES.get(name, name)} | {'выполнено' if value else 'не выполнено'} |"
        for name, value in checks.items()  # type: ignore[union-attr]
    )
    verdict = DECISION_LABELS[str(decision["decision"])]
    delta = float(summary["delta"]["ndcg@3"])  # type: ignore[index]
    improved = int(summary["improved_queries"])
    worsened = int(summary["worsened_queries"])
    dominant = decision["dominant_scenario"]
    reproduction = summary.get("reproduction")
    ceiling = reproduction.get("ceiling") if isinstance(reproduction, dict) else None
    ceiling = ceiling if isinstance(ceiling, dict) else {}
    top10 = ceiling.get("top_10")
    top25 = ceiling.get("top_25")
    headroom = (
        float(top10) - float(summary["baseline"]["ndcg@3"])  # type: ignore[index]
        if top10 is not None
        else 0.0
    )
    used_share = delta / headroom if headroom > 0 else 0.0
    pool_sentence = (
        f"потолок перестановки внутри top-10 — {format_value(float(top10))} из работы 8."
        if top10 is not None
        else "потолок перестановки внутри top-10 ограничен работой 8."
    )
    lines = [
        "# Интерпретация и решение",
        "",
        f"Решение E01: **{verdict}**. {capitalise(str(decision['reason']))}.",
        "",
        f"Главная метрика: NDCG@3 {format_value(float(summary['baseline']['ndcg@3']))} "  # type: ignore[index]
        f"→ {format_value(float(summary['candidate']['ndcg@3']))}, прирост "  # type: ignore[index]
        f"{format_value(delta)} при пороге {format_value(ALPHA_MARGIN, 2)}.",
        "",
        "## Проверка условий, заданных до прогона",
        "",
        "| Условие | Результат |",
        "|---|---|",
        checks_block,
        "",
        "Условия взяты из карточки гипотезы работы 8: прирост не меньше 0,10, "
        "защитные показатели не ниже baseline, пустых выдач ноль, прирост не держится "
        "на одном сценарии (не больше половины общего прироста). Порог не менялся "
        "после прогона.",
        "",
        "## Что показывает результат",
        "",
        f"Порядок изменился у {summary['changed_queries']} запросов из "  # type: ignore[index]
        f"{format_int(summary['scored_queries'])}: лучше стало у {improved}, хуже — у {worsened}. "  # type: ignore[index]
        + (
            f"Сценарий {dominant} даёт {format_share(float(decision['dominant_share']))} "
            "общего прироста; остальное распределено по остальным сценариям."
            if dominant
            else "Ни один сценарий не дал прироста."
        ),
        "",
        signal_sentence(summary),
        "",
        saturation_sentence(summary),
        "",
        "Переранжирование работает там, где карточка той же причины уже попала в пул "
        "top-10, но стояла ниже. Оно не помогает запросам, у которых в пуле нет ни "
        "одной карточки той же причины, и не может поднять карточку с ранга ниже 10: "
        + pool_sentence,
        "",
    ]
    if test_summary is not None:
        test_delta = float(test_summary["delta"]["ndcg@3"])  # type: ignore[index]
        lines += [
            "## Контроль на test",
            "",
            "После фиксации решения кандидат проверен на test: NDCG@3 baseline {base}, "
            "кандидат {cand}, прирост {delta}. Test не участвовал в выборе и решение "
            "не меняет. {verdict}".format(
                base=format_value(float(test_summary["baseline"]["ndcg@3"])),  # type: ignore[index]
                cand=format_value(float(test_summary["candidate"]["ndcg@3"])),  # type: ignore[index]
                delta=format_value(test_delta),
                verdict=(
                    "На других сценариях прирост не подтвердился: это ожидаемо для "
                    "контроля, но означает, что переносить вывод на новые шаблоны нельзя."
                    if test_delta < 0
                    else "Знак прироста на test совпал, но выбор всё равно сделан на validation: "
                    "test — контроль, а не второй отбор."
                ),
            ),
            "",
        ]
    lines += [
        "## Границы вывода",
        "",
        "- Данные синтетические: шаблоны описаний повторяются, поэтому эффект измеряет "
        "согласованность разметки и словаря, а не способность понимать реальные инциденты.",
        "- Людей в измерении нет: ценность для дежурного и время до выбора этой "
        "таблицей не проверяются.",
        "- " + p1_boundary(summary),
        "- Test — контроль после решения, а не второе независимое исследование: "
        "он сгенерирован тем же генератором и делит с train часть словаря, но ни один "
        "шаблон описаний не встречается в двух выборках (проверка работы 5). Поэтому "
        "test контролирует перенос на другие сценарии, а не на естественный язык.",
        "- Гипотеза бинарна: зона 0,05–0,10 заранее объявлена отказом, поэтому "
        "«проверить дополнительно» в этом решении не используется.",
        "",
        "## Следующий шаг",
        "",
    ]
    if decision["decision"] == "include":
        lines += [
            "Изменение включается в alpha как режим по умолчанию, baseline остаётся "
            "fallback. Следующая проверка — на естественных запросах без шаблонов.",
        ]
    else:
        lines += [
            "Изменение в alpha не включается: baseline остаётся основным режимом, "
            "код переранжировщика сохраняется как выключенный режим с тестами.",
            "",
            "Следующий эксперимент должен менять другой фактор — ширину пула. "
            + (
                f"Потолок перестановки внутри top-10 — {format_value(float(top10))}, "
                f"достигнутый прирост занимает {used_share * 100:.0f} % этого запаса"
                if top10 is not None
                else "Потолок перестановки внутри top-10 конечен"
            )
            + (
                f"; в пуле top-25 потолок {format_value(float(top25))}, поэтому "
                "расширение пула проверяется отдельно, а не вместе с сигналом."
                if top25 is not None
                else "; расширение пула проверяется отдельно, а не вместе с сигналом."
            ),
            "",
            "Ответ на вопрос самопроверки: если главная метрика не изменится, а "
            "защитная ухудшится, изменение не включается, потому что чужой runbook "
            "в критичном инциденте дороже прироста средней метрики.",
        ]
    return "\n".join(lines) + "\n"


def render_docs(results_dir: Path, docs_dir: Path) -> None:
    """Пересобирает документы из машинных файлов: числа в тексте не пишутся руками."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"нет сводки {summary_path}: сначала запустите run")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(results_dir / "per_query.csv", dtype={"incident_id": str})
    # Пустая выдача записывается в CSV пустой строкой, а читается как NaN. Текстовые
    # колонки возвращаются к строкам, иначе разбиение по режиму сравнивает float и str.
    text_columns = (
        "incident_id",
        "scenario_id",
        "event_id",
        "language",
        "severity",
        "service",
        "root_cause",
        "ids_base",
        "ids_cand",
        "grades_base",
        "grades_cand",
        "regime",
        "signal",
        "predicted_cause",
    )
    for column in text_columns:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    decision = None
    decision_path = results_dir / "decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    test_summary = None
    test_path = results_dir / "test_summary.json"
    if test_path.exists():
        test_summary = json.loads(test_path.read_text(encoding="utf-8"))

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "comparison.md").write_text(
        render_comparison(summary, frame, test_summary), encoding="utf-8"
    )
    (docs_dir / "experiment_log.md").write_text(
        render_log(summary, decision, test_summary), encoding="utf-8"
    )
    decision_doc = docs_dir / "decision.md"
    if decision is None:
        # Снятое решение не должно оставаться в документах прошлого эксперимента.
        decision_doc.unlink(missing_ok=True)
    else:
        decision_doc.write_text(
            render_decision(summary, decision, test_summary), encoding="utf-8"
        )


PREVIOUS_SUFFIX = ".previous"


def archive_previous(results_dir: Path) -> list[str]:
    """Отодвигает артефакты прошлого эксперимента в `.previous`, а не стирает их.

    `run --force` начинает новый эксперимент, но след прошлого решения и контроля
    должен остаться: иначе повторный выбор после просмотра test нечем поймать.
    """
    archived = []
    for name in ("decision.json", "test_summary.json", "test_per_query.csv"):
        path = results_dir / name
        if not path.exists():
            continue
        target = path.with_name(path.stem + PREVIOUS_SUFFIX + path.suffix)
        target.unlink(missing_ok=True)
        path.replace(target)
        archived.append(target.name)
    return archived


def run(results_dir: Path = RESULTS_DIR, docs_dir: Path | None = None, force: bool = False) -> int:
    """Прогон на validation: baseline и кандидат, без чтения test."""
    docs_dir = docs_dir or results_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    decision_path = results_dir / "decision.json"
    if decision_path.exists() and not force:
        raise SystemExit(
            "решение уже зафиксировано: новый прогон — новый эксперимент. "
            "Удалите decision.json и test_summary.json или запустите с --force"
        )

    frames = load_selected_splits(["train", "validation"])
    train, validation = frames["train"], frames["validation"]
    vectorizer = fit_vectorizer(train)
    matrix = transform(vectorizer, train)
    vocabulary = vocabulary_hash(vectorizer)
    check_manifest(MANIFEST_PATH, vocabulary)
    baseline = RetrievalBaseline(vectorizer, train, matrix)
    reranker = build_reranker(train, matrix)

    frame = evaluate_frame(baseline, reranker, validation)
    summary = build_summary(frame, sorted(train["root_cause"].unique()))
    summary["vocabulary_sha256"] = vocabulary
    summary["run_date"] = date.today().isoformat()
    summary["train_cause_share"] = {
        str(cause): round(float(share), 6)
        for cause, share in train["root_cause"].value_counts(normalize=True).items()
    }

    archive_previous(results_dir)
    frame.to_csv(results_dir / "per_query.csv", index=False)
    segment_table(frame).to_csv(results_dir / "segments.csv", index=False)
    write_json(results_dir / "config.json", summary["config"])
    write_json(results_dir / "summary.json", summary)
    write_json(results_dir / "environment.json", summary["environment"])
    render_docs(results_dir, docs_dir)

    checks = summary["decision_checks"]
    print(f"запросов: {summary['queries']}, в средних: {summary['scored_queries']}")
    print(
        "NDCG@3: baseline {base}, кандидат {cand}, прирост {delta}".format(
            base=format_value(summary["baseline"]["ndcg@3"]),
            cand=format_value(summary["candidate"]["ndcg@3"]),
            delta=format_value(summary["delta"]["ndcg@3"]),
        )
    )
    print(f"порог H1: {format_value(ALPHA_MARGIN)}; условие прироста: {checks['checks']['gain']}")
    print(
        "защитные: P1 причина {base} -> {cand}, P1 NDCG {nb} -> {nc}, пустых {empty}".format(
            base=format_value(float(summary["protective"]["p1_cause_hit_base"])),
            cand=format_value(float(summary["protective"]["p1_cause_hit_cand"])),
            nb=format_value(float(summary["protective"]["p1_ndcg_base"])),
            nc=format_value(float(summary["protective"]["p1_ndcg_cand"])),
            empty=summary["protective"]["empty_cand"],
        )
    )
    print(f"решение по условиям: {expected_decision(checks['checks'])}; зафиксируйте его командой freeze")
    return 0


def freeze(
    decision: str,
    reason: str,
    results_dir: Path = RESULTS_DIR,
    docs_dir: Path | None = None,
    run_date: str | None = None,
) -> int:
    """Фиксирует решение по условиям; несовпадение с правилом — ошибка."""
    docs_dir = docs_dir or results_dir.parent
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"нет сводки {summary_path}: сначала запустите run")
    if decision not in DECISION_LABELS:
        raise SystemExit(f"решение должно быть одним из {sorted(DECISION_LABELS)}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = (summary.get("decision_checks") or {}).get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CONDITION_TITLES):
        raise SystemExit(
            "в сводке нет полного набора условий H1: фиксировать решение не по чему. "
            "Пересоберите сводку командой run"
        )
    if not all(isinstance(value, bool) for value in checks.values()):
        raise SystemExit("условия H1 в сводке должны быть булевыми: сводка повреждена")
    expected = expected_decision(checks)
    if expected != decision:
        raise SystemExit(
            f"решение {decision} противоречит условиям H1: по правилу получается {expected}. "
            "Изменить правило после прогона нельзя; новый порог — новая гипотеза"
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "version": EXPERIMENT_VERSION,
        "date": run_date or date.today().isoformat(),
        "decision": decision,
        "reason": reason,
        "checks": checks,
        "rules": summary["decision_checks"]["rules"],
        "dominant_scenario": summary["decision_checks"]["dominant_scenario"],
        "dominant_share": summary["decision_checks"]["dominant_share"],
        "baseline_ndcg": summary["baseline"]["ndcg@3"],
        "candidate_ndcg": summary["candidate"]["ndcg@3"],
        "delta_ndcg": summary["delta"]["ndcg@3"],
        "summary_sha256": sha256_file(summary_path),
    }
    write_json(results_dir / "decision.json", payload)
    render_docs(results_dir, docs_dir)
    print(f"решение зафиксировано: {decision}; прирост {format_value(payload['delta_ndcg'])}")
    return 0


def confirm_test(results_dir: Path = RESULTS_DIR, docs_dir: Path | None = None) -> int:
    """Контроль на test после решения; до freeze не запускается."""
    docs_dir = docs_dir or results_dir.parent
    summary_path = results_dir / "summary.json"
    decision_path = results_dir / "decision.json"
    if not decision_path.exists():
        raise SystemExit("test запрещён до фиксации решения: сначала freeze")
    if not summary_path.exists():
        raise SystemExit(f"нет сводки {summary_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    recorded = decision.get("summary_sha256")
    if not recorded:
        raise SystemExit("decision.json без отпечатка сводки: test не может служить контролем")
    if sha256_file(summary_path) != recorded:
        raise SystemExit("validation-сводка изменилась после фиксации решения: test не контроль")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = (summary.get("decision_checks") or {}).get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CONDITION_TITLES):
        raise SystemExit("в сводке неполный набор условий H1: test не контроль")
    if expected_decision(checks) != decision.get("decision"):
        raise SystemExit("decision.json не соответствует условиям сводки: test не контроль")
    frames = load_splits()
    train, test = frames["train"], frames["test"]
    vectorizer = fit_vectorizer(train)
    matrix = transform(vectorizer, train)
    vocabulary = vocabulary_hash(vectorizer)
    if vocabulary != summary.get("vocabulary_sha256"):
        raise SystemExit("словарь разошёлся со сводкой validation: test не контроль")
    baseline = RetrievalBaseline(vectorizer, train, matrix)
    reranker = build_reranker(train, matrix)

    frame = evaluate_frame(baseline, reranker, test)
    test_summary = summarise(frame)
    test_summary["config"] = summary["config"]
    test_summary["config_sha256"] = summary["config_sha256"]
    test_summary["hashes"] = input_hashes(["test"])
    test_summary["vocabulary_sha256"] = vocabulary
    test_summary["run_date"] = date.today().isoformat()
    test_summary["decision"] = decision
    write_json(results_dir / "test_summary.json", test_summary)
    frame.to_csv(results_dir / "test_per_query.csv", index=False)
    render_docs(results_dir, docs_dir)

    print(
        "test: NDCG@3 baseline {base}, кандидат {cand}, прирост {delta}".format(
            base=format_value(test_summary["baseline"]["ndcg@3"]),
            cand=format_value(test_summary["candidate"]["ndcg@3"]),
            delta=format_value(test_summary["delta"]["ndcg@3"]),
        )
    )
    print("test использован только как контроль: решение зафиксировано до прогона")
    return 0


def variants(results_dir: Path = RESULTS_DIR) -> int:
    """Сравнивает сигналы сортировки на validation и пишет `variants.json`.

    Это воспроизводимая запись выбора, сделанного до фиксации конфигурации:
    четыре варианта сигнала прогоняются тем же кодом метрик, что и кандидат.
    """
    frames = load_selected_splits(["train", "validation"])
    train, validation = frames["train"], frames["validation"]
    vectorizer = fit_vectorizer(train)
    matrix = transform(vectorizer, train)
    vocabulary = vocabulary_hash(vectorizer)
    check_manifest(MANIFEST_PATH, vocabulary)
    baseline = RetrievalBaseline(vectorizer, train, matrix)
    query_matrix = transform(vectorizer, validation)
    similarity = np.asarray((query_matrix @ matrix.T).toarray())
    causes = train["root_cause"].to_numpy()
    labels = train["root_cause"].astype(str).to_numpy()

    signals: dict[str, float] = {}
    for name, params in (
        ("logreg_balanced", RERANKER_PARAMS),
        ("logreg_plain", {**RERANKER_PARAMS, "class_weight": None}),
    ):
        model = LogisticRegression(**params)
        model.fit(matrix, labels)
        probabilities = model.predict_proba(query_matrix)
        positions = cause_positions(causes, model.classes_)
        signals[name] = mean_ndcg(baseline, validation, similarity, probabilities[:, positions])

    naive = ComplementNB()
    naive.fit(matrix, labels)
    probabilities = naive.predict_proba(query_matrix)
    positions = cause_positions(causes, naive.classes_)
    signals["complement_nb"] = mean_ndcg(
        baseline, validation, similarity, probabilities[:, positions]
    )

    classes = np.array(sorted(set(labels)))
    centroids = np.vstack(
        [np.asarray(matrix[labels == cause].mean(axis=0)) for cause in classes]
    )
    centroid_similarity = np.asarray(query_matrix @ centroids.T)
    positions = cause_positions(causes, classes)
    signals["cause_centroid"] = mean_ndcg(
        baseline, validation, similarity, centroid_similarity[:, positions]
    )

    baseline_ndcg = mean_ndcg(baseline, validation, similarity)
    payload = {
        "version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_split": "validation",
        "queries": int(len(validation)),
        "baseline": baseline_ndcg,
        "candidate": signals["logreg_balanced"],
        "signals": signals,
        "alpha": round(baseline_ndcg + ALPHA_MARGIN, 6),
        "note": "замер сигналов сделан на validation до фиксации конфигурации; "
        "test не использовался",
    }
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_candidate = float(summary["candidate"]["ndcg@3"])  # type: ignore[index]
        matches = abs(signals["logreg_balanced"] - expected_candidate) < 1e-6
        payload["matches_run"] = matches
        if not matches:
            raise ValueError(
                f"кандидат variants {signals['logreg_balanced']} не совпал со сводкой "
                f"{expected_candidate}: метрики посчитаны по-разному"
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "variants.json", payload)

    print(f"baseline: {format_value(baseline_ndcg)}; порог H1: {format_value(payload['alpha'])}")
    for name, value in signals.items():
        print(f"{name}: {format_value(value)}")
    if "matches_run" in payload:
        print(f"кандидат совпадает с прогоном: {payload['matches_run']}")
    return 0


def main() -> int:
    """Разбор аргументов: run, freeze, test."""
    parser = argparse.ArgumentParser(description="ЛР 9: эксперимент с переранжированием")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="прогон на validation")
    run_parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    run_parser.add_argument("--docs-dir", type=Path, default=None)
    run_parser.add_argument("--force", action="store_true", help="новый эксперимент поверх старого")

    freeze_parser = subparsers.add_parser("freeze", help="зафиксировать решение")
    freeze_parser.add_argument("--decision", required=True, choices=sorted(DECISION_LABELS))
    freeze_parser.add_argument("--reason", required=True)
    freeze_parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    freeze_parser.add_argument("--docs-dir", type=Path, default=None)
    freeze_parser.add_argument("--date", default=None)

    test_parser = subparsers.add_parser("test", help="контроль на test после решения")
    test_parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    test_parser.add_argument("--docs-dir", type=Path, default=None)

    variants_parser = subparsers.add_parser(
        "variants", help="сравнить сигналы сортировки на validation"
    )
    variants_parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)

    args = parser.parse_args()
    try:
        if args.command == "run":
            return run(args.results_dir, args.docs_dir, args.force)
        if args.command == "freeze":
            return freeze(args.decision, args.reason, args.results_dir, args.docs_dir, args.date)
        if args.command == "variants":
            return variants(args.results_dir)
        return confirm_test(args.results_dir, args.docs_dir)
    except (FileNotFoundError, ValueError) as error:
        # Так же, как triage_baseline: отсутствие данных — сообщение, а не traceback.
        print(f"Ошибка: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
