"""ЛР 8: спецификация метрик, разрезы baseline и таксономия ошибок.

Модуль не обучает новую модель: он пересчитывает ЛР 7 по шкале релевантности
v2.0 и раскладывает результат по сегментам. Шкала v1.0 из ЛР 7 оставлена рядом
в той же таблице, чтобы было видно, что изменилось и на сколько.

Запуск из корня репозитория:
    python src/eval_report.py run
    python src/eval_report.py case --query-id I000280
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix

from prepare_features import (
    SPLIT_FILES,
    TEXT_COLUMN,
    fit_vectorizer,
    load_splits,
    transform,
    vocabulary_hash,
)
from triage_baseline import (
    TOP_K,
    RetrievalBaseline,
    dcg,
    format_value,
    ideal_grades,
    ndcg,
)

REPORT_VERSION = "1.0"
RESULTS_DIR = Path("docs/lab8/results")
MANIFEST_PATH = Path("docs/lab6/features_manifest.json")

# Шкала v2.0: причина определяет runbook, услуга — владельца. Оценка 2 достижима,
# потому что совпадение причины не требует совпадения услуги.
GRADING_RULE_V2 = (
    "2 — та же root_cause (услуга любая); 1 — та же service при другой причине; "
    "0 — не совпало ни то, ни другое"
)
GRADING_RULE_V1 = (
    "2 — совпали service и root_cause; 1 — совпал service или root_cause; "
    "0 — не совпало ни то, ни другое"
)
# Порог приёмки не подбирается: он равен baseline плюс фиксированный шаг.
ALPHA_MARGIN = 0.10
# Служебные слова и единицы измерения: закрытые классы, поэтому список фиксированный.
# Предлоги, союзы, частицы и артикли не несут метку, но попадают в словарь TF-IDF
# и получают вес, а единицы «шт» и «мин» повторяются во всех описаниях с числами.
STOP_WORDS = frozenset(
    """
    и в во не что он на с со как а то все она так его но да ты к у же вы за бы
    по только ее мне было вот от меня еще нет о из ему теперь когда даже ну ли
    если или для мы их чем была сам чтоб без будто чего раз тоже себе под будет ж
    тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем
    чтобы нее сейчас были куда зачем всех можно при об хоть над надо ней до около
    после перед между через
    the of and are is on in at to for with from by as be was were this that these
    those it its not or if then than when while after before under over within per part
    шт мин час мс
    """.split()
)
# Полосы по доле косинуса top-1, пришедшейся на служебные слова, единицы и числа.
NOISE_BANDS = ((0.30, "noise_low"), (0.50, "noise_mid"))
NOISE_SHARE_THRESHOLD = NOISE_BANDS[1][0]
# Порядок полос в отчёте: по возрастанию шума, а не по алфавиту.
BAND_ORDER = {"signal": ("cause_closer", "signal_tie", "service_closer"), "noise": ("noise_low", "noise_mid", "noise_high")}
# Размеры пула для оценки потолка переранжирования. Потолок top-3 — это перестановка
# уже полученной выдачи; top-10 и top-50 показывают, сколько даёт расширение пула.
ORACLE_POOLS = (3, 10, 25, 50)
# Разница меньше этой считается равенством сигналов: знак такой разницы случаен.
GAP_TIE = 0.01
# Дефекты разметки из ЛР 6: 3 карточки из 150 в пилоте. Числа перенесены,
# потому что работа 8 разметку не проводит и пересчитать их не может.
ANNOTATION_DEFECT = (3, 150)
# Дефекты модуля ЛР 7, найденные специальными тестами: 8 из 8 закрыты.
MODULE_DEFECTS = (8, 8)
LONG_TEXT_PERCENTILE = 0.95

SEGMENT_TITLES = {
    "overall": "Все запросы",
    "signal": "Какой сигнал ближе: причина или услуга",
    "language": "Язык запроса",
    "segment": "Покрытие словаря train",
    "severity": "Серьёзность запроса",
    "scenario_id": "Сценарий запроса",
    "cascade": "Каскад события",
    "noise": "Доля косинуса на служебных словах",
}

PATTERN_LABELS = {
    "2,2,2": "все три карточки той же причины",
    "1,1,1": "все три той же услуги с другой причиной",
    "0,0,0": "ни одной метки",
}


def relevance(query: pd.Series, candidate: pd.Series, scale: str) -> int:
    """Уровень релевантности кандидата: шкала v2 ставит причину выше услуги."""
    same_service = candidate["service"] == query["service"]
    same_cause = candidate["root_cause"] == query["root_cause"]
    if scale == "v2":
        if same_cause:
            return 2
        return 1 if same_service else 0
    if same_service and same_cause:
        return 2
    return 1 if same_service or same_cause else 0


def grade_v2(query: pd.Series, candidate: pd.Series) -> int:
    """Уровень релевантности по шкале ЛР 8: причина важнее услуги."""
    return relevance(query, candidate, "v2")


def grade_v1(query: pd.Series, candidate: pd.Series) -> int:
    """Уровень релевантности по шкале ЛР 7: нужны обе метки сразу."""
    return relevance(query, candidate, "v1")


def ideal_counts(baseline: RetrievalBaseline, query: pd.Series) -> dict[str, tuple[int, int]]:
    """Сколько в индексе карточек каждой оценки для этого запроса.

    Возвращает пары (оценка 2, оценка 1) для шкал v2 и v1. Для v2 оценка 2 —
    любая карточка той же причины; для v1 нужна пара «услуга + причина»,
    которой в этом разбиении нет.
    """
    train = baseline.train
    same_cause = train["root_cause"] == query["root_cause"]
    same_service = train["service"] == query["service"]
    both = int((same_cause & same_service).sum())
    cause = int(same_cause.sum())
    service = int(same_service.sum())
    return {"v2": (cause, service - both), "v1": (both, cause + service - 2 * both)}


def signal_band(gap: float) -> str:
    """Какой сигнал ближе: причина, услуга или они почти равны."""
    if abs(gap) <= GAP_TIE:
        return "signal_tie"
    return "cause_closer" if gap > 0 else "service_closer"


def noise_band(share: float) -> str:
    """Сегмент по доле служебных слов в косинусе: до 30 %, до 50 %, выше."""
    if share <= NOISE_BANDS[0][0]:
        return NOISE_BANDS[0][1]
    if share <= NOISE_BANDS[1][0]:
        return NOISE_BANDS[1][1]
    return "noise_high"


def query_vector(baseline: RetrievalBaseline, text: str) -> csr_matrix:
    """Вектор запроса по неизменному словарю train; пустой текст даёт нулевой вектор."""
    return baseline.vectorizer.transform([text if isinstance(text, str) else ""])


def similarity_scores(baseline: RetrievalBaseline, vector: csr_matrix) -> np.ndarray:
    """Косинус запроса со всем индексом: одно умножение на запрос, а не шесть."""
    if vector.nnz == 0:
        return np.zeros(len(baseline.train))
    return np.asarray((vector @ baseline.matrix.T).toarray()).ravel()


def ranked_ids(baseline: RetrievalBaseline, scores: np.ndarray, k: int) -> list[str]:
    """Top-k идентификаторов по убыванию сходства; при равенстве — по возрастанию ID.

    Правило повторяет `RetrievalBaseline.search`, чтобы потолок и выдача считались
    по одному порядку; совпадение проверяется тестом.
    """
    if k <= 0:
        return []
    candidates = np.flatnonzero(scores > 0)
    if candidates.size == 0:
        return []
    ids = baseline.train["incident_id"].to_numpy()[candidates]
    order = np.lexsort((ids, -scores[candidates]))
    return [str(value) for value in ids[order[:k]]]


@lru_cache(maxsize=8)
def feature_names(vectorizer) -> np.ndarray:
    """Названия признаков: считаются один раз на векторизатор, а не на каждый запрос.

    Кэш держит сам векторизатор, поэтому `id` не переиспользуется сборщиком мусора
    и соседний прогон не получит чужие признаки.
    """
    return vectorizer.get_feature_names_out()


def contribution_parts(
    baseline: RetrievalBaseline, vector: csr_matrix, scores: np.ndarray, incident_id: str
) -> float:
    """Доля косинуса top-1, пришедшаяся на служебные слова, единицы и числа.

    Косинус равен скалярному произведению нормированных векторов, а значит сумме
    вкладов отдельных признаков; доля считается по всем признакам карточки.
    """
    train = baseline.train
    matches = np.flatnonzero((train["incident_id"] == incident_id).to_numpy())
    if vector.nnz == 0 or scores.size == 0 or matches.size == 0:
        return 0.0
    product = vector.multiply(baseline.matrix[int(matches[0])]).tocoo()
    total = float(product.data.sum())
    if total <= 0:
        return 0.0
    features = feature_names(baseline.vectorizer)
    noise = float(
        sum(
            value
            for value, index in zip(product.data, product.col)
            if features[index].isdigit() or features[index] in STOP_WORDS
        )
    )
    return noise / total


def label_rank(baseline: RetrievalBaseline, scores: np.ndarray, field: str, value: str) -> int:
    """Позиция лучшей карточки с нужной меткой в полной выдаче; 0 — таких нет."""
    mask = (baseline.train[field] == value).to_numpy()
    if not mask.any():
        return 0
    best = float(scores[mask].max())
    if best <= 0:
        return 0
    return int((scores > best).sum() + 1)


def neighbour_signals(
    baseline: RetrievalBaseline, scores: np.ndarray, query: pd.Series
) -> tuple[float, float]:
    """Ближайшее сходство по одной услуге без причины и по одной причине без услуги.

    Если услуга ближе, три места в выдаче занимают соседи по услуге, и причина
    не находится; разница этих двух чисел и есть измеряемый механизм отказа.
    """
    train = baseline.train
    same_service = ((train["service"] == query["service"]) & (train["root_cause"] != query["root_cause"])).to_numpy()
    same_cause = ((train["root_cause"] == query["root_cause"]) & (train["service"] != query["service"])).to_numpy()
    service_max = float(scores[same_service].max()) if same_service.any() else 0.0
    cause_max = float(scores[same_cause].max()) if same_cause.any() else 0.0
    return service_max, cause_max


def oracle_ceiling(
    baseline: RetrievalBaseline, scores: np.ndarray, query: pd.Series, pool: int, counts: tuple[int, int]
) -> float:
    """Потолок переранжирования: NDCG@3, если из top-pool идеально взять три оценки.

    Это не результат системы, а верхняя граница для любого переранжировщика,
    который видит только эти pool кандидатов: оценка сверху, а не достижимое значение.
    """
    hits = ranked_ids(baseline, scores, pool)
    grades = [grade_v2(query, baseline.labels(incident_id)) for incident_id in hits]
    return ndcg(sorted(grades, reverse=True), ideal_grades(*counts, TOP_K), TOP_K)


def evaluate(baseline: RetrievalBaseline, queries: pd.DataFrame) -> pd.DataFrame:
    """Один проход поиска на запрос, две шкалы и признаки для разрезов."""
    event_sizes = queries.groupby("event_id").size()
    rows = []
    for _, query in queries.iterrows():
        result = baseline.search(query[TEXT_COLUMN])
        vector = query_vector(baseline, query[TEXT_COLUMN])
        scores = similarity_scores(baseline, vector)
        candidates = [baseline.labels(hit.incident_id) for hit in result.hits]
        grades_v2 = [grade_v2(query, row) for row in candidates]
        grades_v1 = [grade_v1(query, row) for row in candidates]
        counts = ideal_counts(baseline, query)
        counts_v2, counts_v1 = counts["v2"], counts["v1"]
        presented = len(result.hits)

        def recall_of(grades: list[int], pair: tuple[int, int]) -> int:
            """Попадание: в top-k есть карточка с оценкой не ниже 1."""
            return int(any(grade >= 1 for grade in grades) and (pair[0] + pair[1]) > 0)

        def mrr_of(grades: list[int]) -> float:
            first = next((rank for rank, grade in enumerate(grades, start=1) if grade >= 1), None)
            return round(1 / first, 6) if first else 0.0

        top1_id = result.hits[0].incident_id if result.hits else ""
        noise_share = contribution_parts(baseline, vector, scores, top1_id) if top1_id else 0.0
        service_max, cause_max = neighbour_signals(baseline, scores, query)
        rows.append(
            {
                "incident_id": query["incident_id"],
                "scenario_id": query["scenario_id"],
                "event_id": query["event_id"],
                "language": query["language"],
                "severity": query["severity"],
                "service": query["service"],
                "root_cause": query["root_cause"],
                "coverage": round(result.coverage, 6),
                "segment": result.segment,
                "returned": presented,
                "top1_id": top1_id,
                "top1_score": round(result.hits[0].score, 6) if result.hits else 0.0,
                "top1_grade_v2": grades_v2[0] if grades_v2 else 0,
                "grades_v2": ",".join(str(grade) for grade in grades_v2),
                "grades_v1": ",".join(str(grade) for grade in grades_v1),
                "cause_exact_hit": int(
                    any(row["root_cause"] == query["root_cause"] for row in candidates)
                ),
                "service_hit": int(any(row["service"] == query["service"] for row in candidates)),
                "ideal_v2": counts_v2[0] + counts_v2[1],
                "ideal_v1_grade_two": counts_v1[0],
                "no_relevant": int(counts_v2[0] + counts_v2[1] == 0),
                "ndcg_v2": round(ndcg(grades_v2, ideal_grades(*counts_v2, TOP_K), TOP_K), 6),
                "recall_v2": recall_of(grades_v2, counts_v2),
                "mrr_v2": mrr_of(grades_v2),
                "ndcg_v1": round(ndcg(grades_v1, ideal_grades(*counts_v1, TOP_K), TOP_K), 6),
                "cause_rank": label_rank(baseline, scores, "root_cause", query["root_cause"]),
                "noise_share": round(noise_share, 6),
                "noise": noise_band(noise_share),
                "service_max": round(service_max, 6),
                "cause_max": round(cause_max, 6),
                "gap": round(cause_max - service_max, 6),
                "signal": signal_band(cause_max - service_max),
                **{
                    f"ceiling_{pool}": round(oracle_ceiling(baseline, scores, query, pool, counts_v2), 6)
                    for pool in ORACLE_POOLS
                },
                "cascade": "multi" if int(event_sizes[query["event_id"]]) > 1 else "single",
            }
        )
    return pd.DataFrame(rows)


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    """Запросы, у которых в индексе есть хоть одна релевантная карточка.

    Запрос без релевантных объектов исключается из средних: ноль смешал бы
    «система не нашла» и «искать было нечего». Их число считается отдельно.
    """
    return frame[frame["ideal_v2"] > 0]


def aggregate(frame: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
    """Средние по всему набору или по значениям одной колонки.

    Запросы без релевантных объектов в средние не входят; сколько их отброшено,
    отчёт показывает отдельной строкой (см. `excluded`).
    """
    excluded = int(len(frame) - len(usable(frame)))
    frame = usable(frame)

    def summary(group: pd.DataFrame, label: str) -> dict[str, object]:
        if group.empty:
            return {
                "label": label,
                "queries": 0,
                "ndcg_v2": 0.0,
                "recall_v2": 0.0,
                "mrr_v2": 0.0,
                "cause_exact_hit": 0.0,
                "service_hit": 0.0,
                "gap": 0.0,
                "empty": 0,
                "ndcg_v1": 0.0,
            }
        return {
            "label": label,
            "queries": len(group),
            "ndcg_v2": round(float(group["ndcg_v2"].mean()), 6),
            "recall_v2": round(float(group["recall_v2"].mean()), 6),
            "mrr_v2": round(float(group["mrr_v2"].mean()), 6),
            "cause_exact_hit": round(float(group["cause_exact_hit"].mean()), 6),
            "service_hit": round(float(group["service_hit"].mean()), 6),
            "gap": round(float(group["gap"].mean()), 6),
            "empty": int((group["returned"] == 0).sum()),
            "ndcg_v1": round(float(group["ndcg_v1"].mean()), 6),
        }

    if column is None:
        table = pd.DataFrame([summary(frame, "Все запросы")])
    else:
        order = list(BAND_ORDER.get(column, ()))
        values = [value for value in order if value in set(frame[column])]
        values += sorted(set(frame[column]) - set(values))
        rows = [summary(frame[frame[column] == value], str(value)) for value in values]
        table = pd.DataFrame([summary(frame, "Все запросы")] + rows)
    table.attrs["excluded"] = excluded
    return table


def error_taxonomy(frame: pd.DataFrame, summary: dict[str, object] | None = None) -> pd.DataFrame:
    """Типы ошибок с числителем и знаменателем; пример — первый по ID.

    Типы пересекаются: один запрос попадает и в «причина не найдена», и в
    «ни услуги, ни причины». Складывать строки нельзя.
    """
    summary = summary or {}
    total = len(frame)
    p1 = frame[frame["severity"] == "P1"]

    def example(mask: pd.Series) -> str:
        picked = frame[mask].sort_values("incident_id")
        if picked.empty:
            return "—"
        row = picked.iloc[0]
        return f"{row['incident_id']} ({row['grades_v2']})"

    rows = [
        {
            "type": "Причина не найдена в top-3",
            "count": int((frame["cause_exact_hit"] == 0).sum()),
            "denominator": total,
            "example": example(frame["cause_exact_hit"] == 0),
            "scope": "основная выборка",
            "cause": "нет карточки той же причины среди трёх ближайших по косинусу",
            "next": "проверить, чем заняты три позиции: услуга, шаблон, числа",
        },
        {
            "type": "Причина найдена, но не первой",
            "count": int(((frame["top1_grade_v2"] != 2) & (frame["cause_exact_hit"] == 1)).sum()),
            "denominator": total,
            "example": example((frame["top1_grade_v2"] != 2) & (frame["cause_exact_hit"] == 1)),
            "scope": "основная выборка",
            "cause": "карточка той же услуги с другой причиной весит больше нужной",
            "next": "проверить порядок, а не полноту выдачи",
        },
        {
            "type": "Ни услуги, ни причины",
            "count": int((frame["grades_v2"] == "0,0,0").sum()),
            "denominator": total,
            "example": example(frame["grades_v2"] == "0,0,0"),
            "scope": "основная выборка",
            "cause": "совпали только служебные слова и числа",
            "next": "разобрать вклад отдельных слов у этих запросов",
        },
        {
            "type": "Причина не найдена у P1",
            "count": int((p1["cause_exact_hit"] == 0).sum()),
            "denominator": len(p1),
            "example": example((frame["severity"] == "P1") & (frame["cause_exact_hit"] == 0)),
            "scope": "основная выборка, сегмент P1",
            "cause": "то же, но цена ошибки выше: дежурный берёт чужой runbook",
            "next": "считать как защитную метрику, а не как среднее",
        },
        {
            "type": "Причина дальше, чем услуга",
            "count": int((frame["gap"] < -GAP_TIE).sum()),
            "denominator": total,
            "example": example(frame["gap"] < -GAP_TIE),
            "scope": "основная выборка, механизм",
            "cause": "сходство с карточками той же услуги стабильно выше, чем с карточками той же причины",
            "next": "проверить переранжирование пула, а не веса слов",
        },
        {
            "type": "Сходство top-1 больше чем наполовину на служебных словах",
            "count": int((frame["noise_share"] > NOISE_SHARE_THRESHOLD).sum()),
            "denominator": total,
            "example": example(frame["noise_share"] > NOISE_SHARE_THRESHOLD),
            "scope": "основная выборка",
            "cause": "служебные слова и единицы «шт/мин» повторяются, а числа уникальны, "
            "но похожи между собой по единицам измерения",
            "next": "убрать их из текста перед тем же TF-IDF; с качеством мера пока не связана",
        },
        {
            "type": "Пустая выдача",
            "count": int((frame["returned"] == 0).sum()),
            "denominator": total,
            "example": "—",
            "scope": "основная выборка",
            "cause": "нулевой вектор возможен только для пустого запроса; в validation таких нет",
            "next": "держать правило нуля в спецификации и в тестах",
        },
        {
            "type": "Запрос без релевантных объектов",
            "count": int(frame["no_relevant"].sum()),
            "denominator": total,
            "example": "—",
            "scope": "основная выборка, правило учёта",
            "cause": "в индексе нет ни карточки той же причины, ни карточки той же услуги",
            "next": "исключается из средних и считается отдельно; в 1.1 таких нет",
        },
        {
            "type": "Оценка 2 недостижима по шкале v1",
            "count": int((frame["ideal_v1_grade_two"] == 0).sum()),
            "denominator": total,
            "example": "все запросы validation",
            "scope": "ошибка данных и метрики, не модели",
            "cause": "у train и validation нет общих пар «услуга + причина»",
            "next": "закрыто шкалой v2 в этой работе",
        },
        {
            "type": "Дефект текста IMS-02",
            "count": ANNOTATION_DEFECT[0],
            "denominator": ANNOTATION_DEFECT[1],
            "example": "карточки SC-IMS-02, где impact описывает симптом",
            "scope": "разметка, пилот ЛР 6",
            "cause": "в impact попала фраза о росте таймаутов вместо последствия",
            "next": "правка в версии данных 1.2, не сейчас",
        },
        {
            "type": "Дефекты модуля, найденные тестами",
            "count": MODULE_DEFECTS[0],
            "denominator": MODULE_DEFECTS[1],
            "example": "пустой DataFrame в aggregate, NaN в JSON, отрицательный k",
            "scope": "специальные тесты, вне выборки",
            "cause": "граничные случаи не были продуманы при первом написании",
            "next": "закрыто в ЛР 7, регрессия держится тестами",
        },
    ]
    return pd.DataFrame(rows)


def pick_manual_case(frame: pd.DataFrame) -> pd.Series:
    """Запрос для ручного расчёта: оценки 2, 1, 0 в этом порядке."""
    exact = frame[frame["grades_v2"] == "2,1,0"].sort_values("incident_id")
    if exact.empty:
        raise ValueError("нет запроса с оценками 2, 1, 0: ручной расчёт нужно переписать")
    return exact.iloc[0]


def manual_case(
    baseline: RetrievalBaseline, frame: pd.DataFrame, queries: pd.DataFrame, query_id: str | None
) -> dict[str, object]:
    """Данные для ручного расчёта: оценки, вклады и оба значения метрики."""
    if query_id is not None and (frame["incident_id"] == query_id).sum() == 0:
        raise ValueError(f"нет запроса {query_id} в выборке отчёта")
    row = frame[frame["incident_id"] == query_id].iloc[0] if query_id else pick_manual_case(frame)
    if row["incident_id"] not in set(queries["incident_id"]):
        raise ValueError(f"нет запроса {row['incident_id']} в выборке запросов")
    query = queries.set_index("incident_id").loc[row["incident_id"]]
    result = baseline.search(query[TEXT_COLUMN], TOP_K)
    grades = [grade_v2(query, baseline.labels(hit.incident_id)) for hit in result.hits]
    counts = ideal_counts(baseline, query)["v2"]
    ideal = ideal_grades(*counts, TOP_K)
    return {
        "incident_id": row["incident_id"],
        "text": str(query[TEXT_COLUMN]),
        "service": str(query["service"]),
        "root_cause": str(query["root_cause"]),
        "severity": str(query["severity"]),
        "hits": [
            {
                "rank": rank,
                "incident_id": hit.incident_id,
                "score": round(hit.score, 6),
                "grade": grade,
            }
            for rank, (hit, grade) in enumerate(zip(result.hits, grades), start=1)
        ],
        "grades": grades,
        "ideal_grades": ideal,
        "dcg": round(dcg(grades, TOP_K), 5),
        "idcg": round(dcg(ideal, TOP_K), 5),
        "ndcg": round(ndcg(grades, ideal, TOP_K), 6),
        "mrr": float(row["mrr_v2"]),
        "counts": {"grade_two": counts[0], "grade_one": counts[1]},
    }


format_number = format_value


def table_markdown(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    """Markdown-таблица из кадра с заданным порядком колонок.

    Значения с вертикальной чертой экранируются: иначе таблица разваливается.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, row in frame.iterrows():
        cells = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


SEGMENT_COLUMNS = ["label", "queries", "ndcg_v2", "recall_v2", "mrr_v2", "cause_exact_hit", "service_hit", "gap", "empty", "ndcg_v1"]
SEGMENT_HEADERS = [
    "Значение",
    "Запросов",
    "NDCG@3",
    "Recall@3",
    "MRR@3",
    "Причина в top-3",
    "Услуга в top-3",
    "Разница сигналов",
    "Пустых",
    "NDCG@3 по шкале ЛР 7",
]


def segment_block(title: str, note: str, frame: pd.DataFrame) -> str:
    """Раздел отчёта: заголовок, пояснение, таблица."""
    formatted = frame.copy()
    for column in ["ndcg_v2", "recall_v2", "mrr_v2", "cause_exact_hit", "service_hit", "gap", "ndcg_v1"]:
        formatted[column] = formatted[column].map(format_number)
    return "\n\n".join([f"## {title}", note, table_markdown(formatted, SEGMENT_COLUMNS, SEGMENT_HEADERS)])


def segment_note(key: str, frame: pd.DataFrame) -> str:
    """Пояснение к разрезу: числа берутся из данных, а не пишутся руками."""
    languages = frame["language"].value_counts()
    scenarios = frame["scenario_id"].value_counts()
    p1 = frame[frame["severity"] == "P1"]
    events = frame.groupby("event_id").size()
    notes = {
        "overall": "Одно среднее по всему набору; ниже видно, где оно скрывает провал.",
        "signal": "Разница между ближайшей карточкой той же причины и ближайшей карточкой той же услуги "
        "с другой причиной. Если причина ближе, TF-IDF находит нужное; если услуга — три позиции "
        "занимают соседи по услуге, и метрика проваливается. Разница меньше "
        f"{format_number(GAP_TIE)} записана как равенство: знак такой разницы случаен.",
        "language": "Язык карточки: "
        + ", ".join(f"{name} {count}" for name, count in languages.items())
        + ". Разрыв между ними — свойство корпуса, а не модели.",
        "segment": "Доля содержательных слов запроса, известных словарю train; правило из ЛР 7.",
        "severity": "Целевая метка P1–P4. "
        + (
            f"Все {len(p1)} карточек P1 принадлежат одному сценарию {p1['scenario_id'].iloc[0]} "
            f"(услуга {p1['service'].iloc[0]}, причина {p1['root_cause'].iloc[0]}), поэтому строка P1 "
            "измеряет знакомство с этой лексикой, а не поведение на серьёзных инцидентах."
            if len(p1) and p1["scenario_id"].nunique() == 1
            else f"Карточек P1: {len(p1)} в {p1['scenario_id'].nunique()} сценариях."
        ),
        "scenario_id": f"{scenarios.size} сценариев, в каждом {scenarios.min()}–{scenarios.max()} карточек; "
        "строки показывают разброс между шаблонами.",
        "cascade": f"Каскад — событие, где больше одной карточки: {int((events > 1).sum())} событий и "
        f"{int(events[events > 1].sum())} карточек. Такие запросы описывают следствия одного сбоя.",
        "noise": "Доля косинуса top-1, пришедшаяся на служебные слова, единицы «шт/мин» и цифры, "
        "по фиксированному списку в модуле. Показывает запросы, где похожесть держится не на содержании.",
    }
    return notes[key]


def render_segments(per_query: pd.DataFrame, summary: dict[str, object]) -> str:
    """Собирает docs/lab8/segments.md из кадра по запросам."""
    if per_query.empty:
        raise ValueError("пустой набор запросов: разрезы считать не из чего")
    blocks = [
        "# Разрезы baseline",
        "Файл собран командой `python src/eval_report.py run`. Число запросов в каждой строке указано явно: "
        "без знаменателя сегмент с 96 запросами и сегмент с 559 выглядят одинаково убедительно. "
        "Столбец NDCG@3 по шкале ЛР 7 оставлен для сверки с опубликованными метриками baseline: "
        "у этих двух чисел разная шкала, сравнивать их между собой нельзя, только видеть, в какую сторону они разошлись.",
    ]

    patterns = per_query["grades_v2"].value_counts()
    named = [pattern for pattern in PATTERN_LABELS if pattern in patterns.index]
    rest = per_query[~per_query["grades_v2"].isin(named)]
    with_label_first = int(rest["grades_v2"].str[0].isin(["1", "2"]).sum())
    rows = [
        {
            "pattern": pattern,
            "meaning": PATTERN_LABELS[pattern],
            "queries": int(patterns[pattern]),
            "share": format_number(int(patterns[pattern]) / len(per_query)),
            "ndcg": format_number(float(per_query[per_query["grades_v2"] == pattern]["ndcg_v2"].mean())),
        }
        for pattern in named
    ]
    rows.append(
        {
            "pattern": "прочие",
            "meaning": f"остальные сочетания; у {with_label_first} из {len(rest)} метка стоит на первом месте",
            "queries": int(len(rest)),
            "share": format_number(len(rest) / len(per_query)),
            "ndcg": format_number(float(rest["ndcg_v2"].mean())) if len(rest) else "—",
        }
    )
    blocks.append(
        "\n\n".join(
            [
                "## Три режима выдачи",
                "Оценки по трём позициям принимают всего несколько значений, потому что идеал у всех запросов "
                "одинаковый: " + str(summary["ideal"]["grades"]) + ". Три первых строки делят "
                f"{len(per_query)} запросов на режимы, и видно, что средняя NDCG@3 описывает смесь трёх разных "
                "систем, а не одну.",
                table_markdown(
                    pd.DataFrame(rows),
                    ["pattern", "meaning", "queries", "share", "ndcg"],
                    ["Оценки top-3", "Что это значит", "Запросов", "Доля", "NDCG@3"],
                ),
            ]
        )
    )

    for column in ["overall", "signal", "scenario_id", "noise", "severity", "language", "segment", "cascade"]:
        table = aggregate(per_query) if column == "overall" else aggregate(per_query, column)
        blocks.append(segment_block(SEGMENT_TITLES[column], segment_note(column, per_query), table))

    band_table = (
        per_query.assign(regime=per_query["grades_v2"].where(per_query["grades_v2"].isin(named), "прочие"))
        .groupby(["noise", "regime"])
        .size()
        .unstack(fill_value=0)
        .reindex([NOISE_BANDS[0][1], NOISE_BANDS[1][1], "noise_high"])
        .fillna(0)
        .reset_index()
    )
    band_table.columns.name = None
    band_table["noise"] = band_table["noise"].map({"noise_low": "до 30 %", "noise_mid": "30–50 %", "noise_high": "выше 50 %"})
    blocks.append(
        "\n\n".join(
            [
                "## Состав режимов внутри полос шума",
                "Полосы шума отличаются по качеству не потому, что служебные слова мешают, а потому что "
                "в них разная доля режимов. Этот разрез показывает, из чего собрана каждая полоса: "
                "сравнивать средние по полосам без него нельзя.",
                table_markdown(
                    band_table,
                    ["noise"] + [name for name in list(PATTERN_LABELS) + ["прочие"] if name in band_table.columns],
                    ["Полоса шума"] + [name for name in list(PATTERN_LABELS) + ["прочие"] if name in band_table.columns],
                ),
            ]
        )
    )

    means = per_query.groupby("scenario_id")["ndcg_v2"].mean()
    p1 = per_query[per_query["severity"] == "P1"]
    blocks.append(
        "\n\n".join(
            [
                "## Разброс по сценариям",
                "Разброс между шаблонами важнее среднего: если минимум близок к нулю, среднее по всем запросам "
                "описывает не систему, а долю удачных шаблонов.",
                "\n".join(
                    [
                        f"- минимум средних: {format_number(float(means.min()))} ({means.idxmin()}), "
                        f"максимум средних: {format_number(float(means.max()))} ({means.idxmax()}); "
                        f"размах {format_number(float(means.max() - means.min()))}",
                        f"- стандартное отклонение средних по сценариям: "
                        f"{format_number(float(means.std()) if len(means) > 1 else 0.0)}",
                        f"- средняя NDCG@3 по всем запросам: {format_number(float(summary['ndcg']['v2']))}",
                        f"- разница сигналов и средняя NDCG@3 по сценариям коррелируют на "
                        f"{format_number(float(summary['gap']['scenario_correlation']))}: там, где причина ближе услуги, "
                        "поиск работает",
                        f"- медиана ранга лучшей карточки той же причины — {summary['cause_rank']['median']}: "
                        "нужное рядом, но ниже отсечения",
                        f"- сегмент P1 целиком лежит в сценарии {p1['scenario_id'].iloc[0]}: "
                        f"все {len(p1)} карточек — услуга {p1['service'].iloc[0]}, причина {p1['root_cause'].iloc[0]}. "
                        "Строка P1 в таблице серьёзности повторяет строку этого сценария, а не измеряет серьёзность.",
                    ]
                ),
            ]
        )
    )

    excluded = int(aggregate(per_query).attrs.get("excluded", 0))
    ceiling_rows = [
        {
            "pool": f"top-{pool}",
            "value": format_number(float(per_query[f"ceiling_{pool}"].mean())),
            "gain": format_number(float(per_query[f"ceiling_{pool}"].mean() - summary["ndcg"]["v2"])),
        }
        for pool in ORACLE_POOLS
    ]
    blocks.append(
        "\n\n".join(
            [
                "## Потолок переранжирования",
                "Сколько дала бы идеальная сортировка, если выбирать три оценки из уже полученной выдачи. "
                "Это не результат системы и не план: это верхняя граница для любого переранжировщика, "
                "который видит только эти кандидаты. Разница с текущей NDCG@3 показывает, есть ли смысл "
                "в переранжировании и на каком пуле.",
                table_markdown(
                    pd.DataFrame(ceiling_rows),
                    ["pool", "value", "gain"],
                    ["Пул", "NDCG@3 сверху", "Прирост к baseline"],
                ),
                "Пул top-3 повторяет текущую выдачу и потолком почти не является: переставить "
                "три карточки можно, только если в них уже есть оценка 2.",
                f"Запросов без релевантных объектов: {excluded} из {len(per_query)}. "
                "Они исключены из средних и из строк этой таблицы; правило записано в спецификации.",
            ]
        )
    )
    return "\n\n".join(blocks)


def render_taxonomy(errors: pd.DataFrame, summary: dict[str, object]) -> str:
    """Собирает docs/lab8/error_taxonomy.md из таблицы ошибок."""

    def count(value: int, denominator: int) -> str:
        share = value / denominator if denominator else 0.0
        return f"{value} из {denominator} ({format_number(share)})"

    lines = [
        "# Таксономия ошибок",
        "Файл собран командой `python src/eval_report.py run`. Частота всегда записана как «числитель из знаменателя»: "
        "одинаковые доли у сегментов в 96 и 559 запросов означают разное.",
        "Типы пересекаются: один запрос может попасть в «причина не найдена» и в «ни услуги, ни причины». "
        "Складывать строки нельзя, сумма больше числа запросов.",
        "## Типы ошибок",
    ]
    table = [
        "| Тип | Частота | Где считано | Пример | Проверенная причина | Следующий шаг |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in errors.iterrows():
        table.append(
            f"| {row['type']} | {count(int(row['count']), int(row['denominator']))} | {row['scope']} | "
            f"{row['example']} | {row['cause']} | {row['next']} |"
        )
    lines.append("\n".join(table))

    def row(kind: str) -> pd.Series:
        return errors[errors["type"] == kind].iloc[0]

    p1 = row("Причина не найдена у P1")
    numeric = row("Сходство top-1 больше чем наполовину на служебных словах")
    gap_row = row("Причина дальше, чем услуга")
    pattern_labels = PATTERN_LABELS
    counts = summary["patterns"]["counts"]
    bullets = [
        "## Что эти цифры позволяют утверждать",
        f"- Главный механизм измерен: у {count(int(gap_row['count']), int(gap_row['denominator']))} запросов "
        f"ближайшая карточка той же услуги ближе, чем ближайшая карточка той же причины, в среднем на "
        f"{format_number(abs(float(summary['gap']['service_closer_mean'])))}. Корреляция этой разницы со средней "
        f"NDCG@3 по сценариям — {format_number(float(summary['gap']['scenario_correlation']))}, по отдельным "
        f"запросам — {format_number(float(summary['gap']['query_correlation']))}.",
        "- Режимы выдачи объясняют среднее лучше, чем модель: "
        + ", ".join(f"«{pattern}» — {counts.get(pattern, 0)}" for pattern in pattern_labels)
        + f". Средняя NDCG@3 — это доля каждого режима, а не одна система.",
        f"- У P1 причина не найдена {count(int(p1['count']), int(p1['denominator']))}. "
        + (
            f"Но все карточки P1 лежат в сценарии {summary['p1']['scenarios'][0]}, где услуга ближе причины на "
            f"{format_number(abs(float(summary['p1']['gap'])))}. Цифра говорит о лексике этого сценария, "
            "а не о серьёзных инцидентах вообще; это ограничение измерения, а не вывод о продукте."
            if summary["p1"]["scenarios"]
            else "Карточек P1 в выборке нет, показатель не определён."
        ),
        f"- Служебные слова и единицы дают больше половины косинуса у {count(int(numeric['count']), int(numeric['denominator']))} "
        "запросов, но доля шума не отделяет провал от успеха: полосы идут 0,297 / 0,561 / 0,298, и разница между "
        "ними объясняется составом режимов, а не шумом (см. `segments.md`). Строка остаётся измеренным, "
        "но не подтверждённым объяснением.",
        f"- Медиана ранга лучшей карточки той же причины — {summary['cause_rank']['median']}: нужное рядом, но не на "
        f"первых местах. Идеальная сортировка пула top-10 даёт NDCG@3 "
        f"{format_number(float(summary['ceiling']['top_10']))}, top-50 — "
        f"{format_number(float(summary['ceiling']['top_50']))}. То есть у переранжирования есть запас, "
        "а у угадывания с первого ранга — нет.",
    ]
    lines.append("\n".join(bullets))

    module = row("Дефекты модуля, найденные тестами")
    annotation = row("Дефект текста IMS-02")
    no_relevant = row("Запрос без релевантных объектов")
    lines.append(
        "\n\n".join(
            [
                "## Ошибки вне основной выборки",
                "Отдельно от модели: это дефекты кода, найденные специальными тестами в ЛР 7, и дефект разметки "
                "из ЛР 6. Они не входят в знаменатель основной выборки, и складывать их с ошибками поиска нельзя.",
                "\n".join(
                    [
                        f"- Пустой запрос, неподдерживаемый k, пустой train, отсутствующий манифест словаря: "
                        f"{int(module['count'])} из {int(module['denominator'])} проверок отклонены или исправлены; "
                        "три из восьми находок меняли опубликованные числа.",
                        f"- Дефект разметки IMS-02: {int(annotation['count'])} из {int(annotation['denominator'])} "
                        "карточек пилота; правка переносится в версию данных 1.2.",
                        f"- Запросы без релевантных объектов: {int(no_relevant['count'])} из "
                        f"{int(no_relevant['denominator'])}. Такой запрос исключается из среднего и считается "
                        "отдельной строкой; это правило проверено тестом на мини-корпусе.",
                    ]
                ),
            ]
        )
    )
    lines.append(
        "\n\n".join(
            [
                "## Чего в этой таблице нет",
                "\n".join(
                    [
                        "- Жалоб пользователей: настоящих дежурных в измерении не было, ущерб оценивается "
                        "по механизму ошибки.",
                        "- Стоимости ложного срабатывания: продуктовая цель «пять минут до выбора» этой "
                        "таблицей не измеряется.",
                        "- Ошибок разметки внутри validation: шкала v2 считается по меткам, а не по экспертным "
                        "оценкам, поэтому ошибка метки воспроизводится в метрике как ошибка поиска.",
                    ]
                ),
            ]
        )
    )
    return "\n\n".join(lines)


def render_manual(case: dict[str, object]) -> str:
    """Собирает docs/lab8/manual_calculation.md под выбранный запрос."""
    hits = case["hits"]
    grades = case["grades"]
    terms = [
        f"{(2 ** int(hit['grade']) - 1)}/{format_number(math.log2(position + 1), 5)} = "
        f"{format_number((2 ** int(hit['grade']) - 1) / math.log2(position + 1), 5)}"
        for position, hit in enumerate(hits, start=1)
    ]
    ideal = case["ideal_grades"]
    ideal_terms = [
        f"{2 ** int(grade) - 1}/{format_number(math.log2(position + 1), 5)}"
        for position, grade in enumerate(ideal, start=1)
    ]
    swapped = [grades[1], grades[0], *grades[2:]] if len(grades) >= 2 else grades
    swapped_value = dcg(swapped, TOP_K)
    table = ["| Ранг | Карточка | Сходство | Оценка |", "|---|---|---|---|"]
    for hit in hits:
        table.append(
            f"| {hit['rank']} | {hit['incident_id']} | {format_number(float(hit['score']))} | {hit['grade']} |"
        )

    lines = [
        "# Ручной расчёт",
        f"Файл собран командой `python src/eval_report.py run`, запрос {case['incident_id']} "
        f"({case['service']}, {case['root_cause']}, {case['severity']}). Машинная копия чисел — "
        "`results/manual_case.json`.",
        "## Запрос и выдача",
        f"> {case['text']}",
        "\n".join(table),
        "## Счёт вручную",
        f"Gain равен 2 в степени оценки минус 1, дисконт позиции равен log2(позиция + 1), позиции считаются "
        f"с единицы. Слагаемые: {'; '.join(terms)}. Сумма — {format_number(float(case['dcg']), 5)}.",
        f"В индексе {case['counts']['grade_two']} карточек с той же причиной и "
        f"{case['counts']['grade_one']} карточек той же услуги с другой причиной, поэтому идеальный порядок — "
        f"{ideal} с полезностью {' + '.join(ideal_terms)} = {format_number(float(case['idcg']), 5)}.",
        f"NDCG@3 = {format_number(float(case['dcg']), 5)} / {format_number(float(case['idcg']), 5)} = "
        f"{format_number(float(case['ndcg']))}. MRR@3 по этому запросу равен {format_number(float(case['mrr']))}.",
        "## Проверка вырожденных случаев",
        f"Если поменять местами первые два результата, оценки станут {swapped}, полезность "
        f"{format_number(swapped_value, 5)}, то есть NDCG {format_number(ndcg(swapped, ideal, TOP_K))}: "
        "порядок наказывается, но не обнуляется.",
        "Пустая выдача даёт NDCG, Recall и MRR, равные нулю, и учитывается отдельным столбцом «Пустых», "
        "а не нулём в среднем. Это проверяется тестом на пустой строке и на `k = 0`, чтобы ошибка "
        "агрегирования не скрылась за средним.",
        "## Сверка с кодом",
        f"Значение из `results/manual_case.json`: NDCG@3 {format_number(float(case['ndcg']))}, "
        f"оценки {grades}. Расхождений нет: `dcg` и `ndcg` берутся из `src/triage_baseline.py`, "
        "тех же функций, что считает прогон, а не переписаны здесь заново.",
    ]
    return "\n\n".join(lines)


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    """Корреляция без NaN: на константе или одном наблюдении она не определена."""
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return 0.0
    value = float(left.corr(right))
    return 0.0 if math.isnan(value) else value


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Пишет JSON с отступами и запретом NaN: формально невалидный JSON лучше падать."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_manifest(path: Path, vocabulary: str) -> None:
    """Сверяет отпечаток словаря с манифестом ЛР 6: признаки не должны меняться незаметно."""
    if not path.exists():
        raise FileNotFoundError(f"нет манифеста {path}: сначала запустите python src/prepare_features.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("vocabulary_sha256")
    if expected != vocabulary:
        raise ValueError(
            f"словарь разошёлся с манифестом ЛР 6: {expected} против {vocabulary}; "
            "признаки пересчитаны или изменены"
        )


def build_summary(
    per_query: pd.DataFrame,
    errors: pd.DataFrame,
    queries: pd.DataFrame,
    vocabulary: str,
    stop_words_present: int,
) -> dict[str, object]:
    """Сводка: главная метрика, порог приёмки и защитные показатели."""
    if per_query.empty:
        raise ValueError("пустой набор запросов: сводку считать не из чего")
    frame = usable(per_query)
    overall_v2 = float(frame["ndcg_v2"].mean()) if len(frame) else 0.0
    overall_v1 = float(frame["ndcg_v1"].mean()) if len(frame) else 0.0
    p1 = frame[frame["severity"] == "P1"]
    p1_cause = float(p1["cause_exact_hit"].mean()) if len(p1) else 0.0
    alpha = round(overall_v2 + ALPHA_MARGIN, 6)
    patterns = per_query["grades_v2"].value_counts()
    largest = patterns.index[0] if len(patterns) else ""
    scenario_gap = per_query.groupby("scenario_id")["gap"].mean()
    scenario_ndcg = per_query.groupby("scenario_id")["ndcg_v2"].mean()
    ideal_pair = ideal_counts_from(per_query)
    return {
        "version": REPORT_VERSION,
        "grading_rules": {"v2": GRADING_RULE_V2, "v1": GRADING_RULE_V1, "active": "v2"},
        "thresholds": {
            "alpha_rule": f"baseline NDCG@3 + {ALPHA_MARGIN}",
            "alpha": alpha,
            "protective_rule": "причина в top-3 у P1 и NDCG@3 у P1 не ниже baseline; пустых выдач 0",
            "protective": {
                "p1_cause_hit": round(p1_cause, 6),
                "p1_ndcg_v2": round(float(p1["ndcg_v2"].mean()), 6) if len(p1) else 0.0,
                "empty": int((per_query["returned"] == 0).sum()),
            },
        },
        "ndcg": {"v2": round(overall_v2, 6), "v1": round(overall_v1, 6), "delta": round(overall_v2 - overall_v1, 6)},
        "recall": {"v2": round(float(frame["recall_v2"].mean()), 6) if len(frame) else 0.0},
        "mrr": {"v2": round(float(frame["mrr_v2"].mean()), 6) if len(frame) else 0.0},
        "queries": int(len(per_query)),
        "scored_queries": int(len(frame)),
        "empty_outputs": int((per_query["returned"] == 0).sum()),
        "short_outputs": int((per_query["returned"] < TOP_K).sum()),
        "no_relevant_queries": int((per_query["ideal_v2"] == 0).sum()),
        "ideal": {
            "uniform": bool((per_query["ideal_v2"] >= TOP_K).all()),
            "grades": [2] * TOP_K,
            "rule": "идеал из трёх оценок 2, потому что у каждого запроса в индексе есть карточки той же причины",
        },
        "p1": {
            "queries": int(len(p1)),
            "scenarios": sorted(p1["scenario_id"].unique().tolist()),
            "services": sorted(p1["service"].unique().tolist()),
            "causes": sorted(p1["root_cause"].unique().tolist()),
            "gap": round(float(p1["gap"].mean()), 6) if len(p1) else 0.0,
        },
        "patterns": {
            "counts": {str(pattern): int(count) for pattern, count in patterns.items()},
            "largest": {"pattern": str(largest), "count": int(patterns.get(largest, 0))},
        },
        "noise": {
            "rule": "доля косинуса top-1 на служебных словах, единицах и цифрах; список в STOP_WORDS",
            "above_50": int((per_query["noise_share"] > NOISE_SHARE_THRESHOLD).sum()),
            "above_30": int((per_query["noise_share"] > NOISE_BANDS[0][0]).sum()),
            "median": round(float(per_query["noise_share"].median()), 6),
            "stop_words_present": stop_words_present,
            "stop_word_list_size": len(STOP_WORDS),
        },
        "gap": {
            "rule": "сходство с ближайшей карточкой той же причины минус сходство с ближайшей "
            "карточкой той же услуги и другой причины",
            "tie": GAP_TIE,
            "cause_closer": int((per_query["gap"] > GAP_TIE).sum()),
            "service_closer": int((per_query["gap"] < -GAP_TIE).sum()),
            "median": round(float(per_query["gap"].median()), 6),
            "service_closer_mean": round(
                float(per_query.loc[per_query["gap"] < -GAP_TIE, "gap"].mean())
                if (per_query["gap"] < -GAP_TIE).any()
                else 0.0,
                6,
            ),
            "query_correlation": round(safe_corr(per_query["gap"], per_query["ndcg_v2"]), 6),
            "scenario_correlation": round(safe_corr(scenario_gap, scenario_ndcg), 6),
        },
        "cause_rank": {
            "rule": "позиция лучшей карточки той же причины в полной выдаче; 0 — таких карточек нет",
            "median": int(per_query["cause_rank"].median()),
            "within_10": round(float((per_query["cause_rank"].between(1, 10)).mean()), 6),
            "within_50": round(float((per_query["cause_rank"].between(1, 50)).mean()), 6),
        },
        "ceiling": {
            f"top_{pool}": round(float(per_query[f"ceiling_{pool}"].mean()), 6) for pool in ORACLE_POOLS
        },
        "ideal_grade_two_by_scale": ideal_pair,
        "errors": {
            row["type"]: {"count": int(row["count"]), "denominator": int(row["denominator"])}
            for _, row in errors.iterrows()
        },
        "vocabulary_hash": vocabulary,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "inputs": SPLIT_FILES,
        "validation_text_length": {
            "median": int(queries[TEXT_COLUMN].astype(str).str.len().median()),
            "p95": int(queries[TEXT_COLUMN].astype(str).str.len().quantile(LONG_TEXT_PERCENTILE)),
        },
    }


def ideal_counts_from(frame: pd.DataFrame) -> dict[str, int]:
    """Сколько запросов имеют в индексе карточку с оценкой 2 по каждой шкале."""
    return {
        "v2": int((frame["ideal_v2"] > 0).sum()),
        "v1": int((frame["ideal_v1_grade_two"] > 0).sum()),
    }


def run(results_dir: Path = RESULTS_DIR, docs_dir: Path | None = None) -> int:
    """Считает отчёты ЛР 8 и пишет их: машинные файлы в `results_dir`, тексты в `docs_dir`.

    По умолчанию тексты идут в родительский каталог результатов, то есть в
    `docs/lab8`, как в работе 7. Если каталог результатов переопределён, тексты
    тоже уходят рядом с ним, а не в отслеживаемый `docs/lab8`.
    """
    docs_dir = docs_dir or results_dir.parent
    frames = load_splits()
    train, validation = frames["train"], frames["validation"]
    vectorizer = fit_vectorizer(train)
    matrix = transform(vectorizer, train)
    vocabulary = vocabulary_hash(vectorizer)
    check_manifest(MANIFEST_PATH, vocabulary)
    baseline = RetrievalBaseline(vectorizer, train, matrix)

    per_query = evaluate(baseline, validation)
    errors = error_taxonomy(per_query)
    stop_words_present = len(set(vectorizer.get_feature_names_out()) & STOP_WORDS)
    summary = build_summary(per_query, errors, validation, vocabulary, stop_words_present)

    results_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(results_dir / "per_query.csv", index=False)
    aggregate(per_query).to_csv(results_dir / "overall.csv", index=False)
    errors.to_csv(results_dir / "errors.csv", index=False)
    write_json(results_dir / "summary.json", summary)
    case = manual_case(baseline, per_query, validation, None)
    write_json(results_dir / "manual_case.json", case)

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "segments.md").write_text(render_segments(per_query, summary), encoding="utf-8")
    (docs_dir / "error_taxonomy.md").write_text(render_taxonomy(errors, summary), encoding="utf-8")
    (docs_dir / "manual_calculation.md").write_text(render_manual(case), encoding="utf-8")

    print(f"запросов: {summary['queries']}, в средних: {summary['scored_queries']}")
    print(f"NDCG@3 v2: {summary['ndcg']['v2']} (v1: {summary['ndcg']['v1']})")
    print(f"Recall@3 v2: {summary['recall']['v2']}")
    print(f"alpha: {summary['thresholds']['alpha']}")
    print(f"пустых выдач: {summary['empty_outputs']}, без релевантных объектов: {summary['no_relevant_queries']}")
    print(f"медиана ранга лучшей карточки той же причины: {summary['cause_rank']['median']}")
    print(
        f"служебные слова: больше половины косинуса у {summary['noise']['above_50']} запросов, "
        f"больше 30 % — у {summary['noise']['above_30']}; в словаре {stop_words_present} служебных токенов"
    )
    return 0


def case_report(query_id: str | None) -> int:
    """Показывает один запрос так же, как его считает отчёт.

    Если запрос задан, считается только он: прогонять все 1 200 строк ради одной
    карточки незачем.
    """
    frames = load_splits()
    vectorizer = fit_vectorizer(frames["train"])
    baseline = RetrievalBaseline(vectorizer, frames["train"], transform(vectorizer, frames["train"]))
    validation = frames["validation"]
    subset = validation[validation["incident_id"] == query_id] if query_id else validation
    if query_id and subset.empty:
        raise SystemExit(f"нет запроса {query_id} в validation")
    per_query = evaluate(baseline, subset)
    case = manual_case(baseline, per_query, validation, query_id)
    print(json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def main() -> int:
    """Разбор аргументов: run или case."""
    parser = argparse.ArgumentParser(description="Отчёты ЛР 8: метрики, разрезы, ошибки")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="пересчитать отчёты")
    run_parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="каталог машинных файлов; тексты пишутся в его родительский каталог",
    )
    case_parser = subparsers.add_parser("case", help="показать один запрос")
    case_parser.add_argument("--query-id", default=None)
    args = parser.parse_args()

    if args.command == "run":
        return run(args.results_dir)
    return case_report(args.query_id)


if __name__ == "__main__":
    raise SystemExit(main())
