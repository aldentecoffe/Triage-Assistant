"""Измерения для архитектурного решения ЛР 10.

ЛР 10 требует, чтобы архитектурное решение было проверяемым: измеренные
оценки отделены от допущений. Поэтому все варианты, которые можно
реализовать локально, сравниваются на одном признаковом пространстве и на
одних и тех же 1 200 карточках validation:

* вариант A — метки как голосование по top-k соседям (сумма косинусов);
* вариант B — три линейных классификатора на том же TF-IDF;
* вариант B-R — критичность по опубликованной таблице последствий
  (`IMPACT_SEVERITY`), то есть по определению метки, а не по модели.

Дополнительно измеряются кривая «покрытие — качество» для порога
уверенности и задержка полного прохода на один инцидент, потому что от них
зависит схема компонентов.

Test не читается: модуль загружает только train и validation.

Запуск из корня репозитория:
    python src/arch_check.py run
    python src/arch_check.py figure
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

from experiment import environment, input_hashes, load_selected_splits
from incident_taxonomy import IMPACT_SEVERITY, ROOT_CAUSES, SCENARIOS, SERVICES, SEVERITIES
from prepare_features import (
    TEXT_COLUMN,
    VECTORIZER_PARAMS,
    fit_vectorizer,
    transform,
    vocabulary_hash,
)
from text_utils import MIN_TOKEN_COUNT, normalise

CHECK_VERSION = "1.0"
RESULTS_DIR = Path("docs/lab10/results")
FIGURE_PATH = Path("docs/lab10/figures/architecture_flow.png")
MANIFEST_PATH = Path("docs/lab6/features_manifest.json")
LAB7_METRICS = Path("docs/lab7/results/retrieval_metrics.json")
LAB8_SUMMARY = Path("docs/lab8/results/summary.json")
LAB9_SUMMARY = Path("docs/lab9/results/summary.json")

TARGETS = ("severity", "service", "root_cause")
# Списки классов берутся из таксономии, а не из train: если класс пропадёт из
# обучающей выборки, F1-macro всё равно считается по полному списку.
TAXONOMY_CLASSES = {
    "severity": sorted(SEVERITIES),
    "service": sorted(SERVICES),
    "root_cause": sorted(ROOT_CAUSES),
}
VOTE_KS = (3, 5, 10)
GATE_THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)
TOP_K = 3
GATE_NOTE = "alpha показывает метку только при уверенности не ниже порога; ниже порога выводится «нужно решение инженера»"
SPLITS_USED = ("train", "validation")


class ArchCheckError(ValueError):
    """Собственная ошибка проверки: отличает её от сбоев sklearn и numpy."""

# Параметры классификаторов объявлены здесь, а не импортируются из ЛР 9: это
# самостоятельное решение ЛР 10. Значения совпадают с переранжировщиком ЛР 9,
# чтобы качество сравнивалось с уже опубликованной моделью, а не с новой.
CLASSIFIER_PARAMS = {
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 2000,
    "class_weight": "balanced",
}

# Ворота взяты из требований проекта (ЛР 1), а не из результата.
LATENCY_BUDGET_MS = 1000.0
PLAN_TARGETS = {
    "severity": 0.75,
    "service": 0.80,
    "root_cause": 0.65,
    "p1_recall": 0.90,
    "p95_latency_ms": LATENCY_BUDGET_MS,
}

# Правило сравнения вариантов фиксируется до запуска, чтобы выбор нельзя было
# подогнать под результат.
SELECTION_RULE = (
    "Для каждой метки берётся вариант с наибольшим F1-macro на validation. "
    "Если разница не больше 0,01, выбирается вариант с меньшим числом "
    "артефактов: гибрид «правило плюс классификатор» проще отдельного "
    "классификатора, классификатор проще голосования по соседям, голосование "
    "проще правила без запасного пути. Test в выборе не участвует. Бюджет "
    "задержки и цели проекта — отдельные ворота: вариант, у которого p95 выше "
    "1 000 мс, не выбирается даже при лучшем F1, а недостижение цели плана "
    "записывается как известное ограничение, а не как повод сменить правило."
)

TOOLS = (
    ("python", None, "интерпретатор; все вычисления выполняются локально"),
    ("numpy", "numpy", "разреженные операции и косинусная близость"),
    ("pandas", "pandas", "чтение выборок и таблиц отчёта"),
    ("scikit-learn", "sklearn", "TF-IDF, LogisticRegression, метрики"),
    ("scipy", "scipy", "разреженные матрицы под scikit-learn"),
    ("joblib", "joblib", "сохранение артефакта модели"),
    ("matplotlib", "matplotlib", "схема компонентов для отчёта"),
    ("fastapi", "fastapi", "локальный HTTP-интерфейс alpha"),
    ("uvicorn", "uvicorn", "запуск FastAPI без внешних сервисов"),
    ("pydantic", "pydantic", "схема запроса и валидация"),
    ("pytest", "pytest", "тесты модуля"),
)


# --- признаки -------------------------------------------------------------


def fit_space(frames: dict[str, pd.DataFrame]):
    """Словарь и IDF только по train; матрицы обеих выборок считаются сразу."""
    vectorizer = fit_vectorizer(frames["train"])
    train_matrix = transform(vectorizer, frames["train"])
    validation_matrix = transform(vectorizer, frames["validation"])
    return vectorizer, train_matrix, validation_matrix


def manifest_check(vectorizer) -> dict[str, object]:
    """Сверяет словарь с манифестом ЛР 6: другой словарь означает другой эксперимент."""
    payload: dict[str, object] = {"path": str(MANIFEST_PATH), "checked": False}
    if not MANIFEST_PATH.exists():
        payload["note"] = "манифест ЛР 6 отсутствует, сверка не выполнена"
        return payload
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual = vocabulary_hash(vectorizer)
    expected = manifest.get("vocabulary_sha256")
    payload.update(
        {
            "checked": True,
            "expected": expected,
            "actual": actual,
            "match": actual == expected,
        }
    )
    return payload


# --- вариант A: голосование по соседям ------------------------------------


def neighbour_order(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы top-k по убыванию близости; нулевые пары отбрасываются."""
    if k <= 0 or scores.size == 0:
        return np.array([], dtype=int)
    limit = min(int(k), int(scores.shape[0]))
    order = np.argsort(-scores, kind="stable")[:limit]
    return order[scores[order] > 0]


def weighted_vote(scores: np.ndarray, labels: np.ndarray, classes: list[str]) -> str | None:
    """Метка соседей: сумма косинусов, ничья — ближайший сосед, затем алфавит."""
    totals = dict.fromkeys(classes, 0.0)
    nearest = dict.fromkeys(classes, 0.0)
    for score, label in zip(scores, labels):
        label = str(label)
        if not np.isfinite(score) or score <= 0 or label not in totals:
            continue
        totals[label] += float(score)
        if float(score) > nearest[label]:
            nearest[label] = float(score)

    candidates = [label for label in classes if totals[label] > 0]
    if not candidates:
        return None
    return sorted(candidates, key=lambda label: (-totals[label], -nearest[label], label))[0]


def vote_all(
    frames: dict[str, pd.DataFrame],
    validation_matrix,
    train_matrix,
    ks: tuple[int, ...] = VOTE_KS,
    support_k: int = TOP_K,
    step: int = 200,
):
    """Один проход по матрице близости: предсказания для всех k и доля веса соседей.

    Матрица `validation @ train.T` считается один раз, а не для каждого k:
    на 1 200 запросах это экономит несколько секунд, и ровно такое дублирование
    уже снимали в ЛР 8.
    """
    train = frames["train"]
    labels = {target: train[target].astype(str).to_numpy() for target in TARGETS}
    classes = TAXONOMY_CLASSES
    predicted = {k: {target: [] for target in TARGETS} for k in ks}
    support: dict[str, list[float]] = {target: [] for target in TARGETS}
    abstained = 0
    limit = max(max(ks), support_k)

    for start in range(0, validation_matrix.shape[0], step):
        block = (validation_matrix[start : start + step] @ train_matrix.T).toarray()
        for row in block:
            order = neighbour_order(row, limit)
            if order.size == 0:
                abstained += 1
                for k in ks:
                    for target in TARGETS:
                        predicted[k][target].append("")
                for target in TARGETS:
                    support[target].append(0.0)
                continue
            scores = row[order]
            for k in ks:
                window = slice(0, min(k, order.size))
                for target in TARGETS:
                    label = weighted_vote(scores[window], labels[target][order[window]], classes[target])
                    predicted[k][target].append(label or "")
            window = slice(0, min(support_k, order.size))
            for target in TARGETS:
                support[target].append(
                    winner_share(scores[window], labels[target][order[window]], classes[target])
                )
    return predicted, support, abstained


def winner_share(scores: np.ndarray, labels: np.ndarray, classes: list[str]) -> float:
    """Доля веса победившей метки среди всех соседей окна."""
    totals = dict.fromkeys(classes, 0.0)
    for score, label in zip(scores, labels):
        label = str(label)
        if score > 0 and label in totals:
            totals[label] += float(score)
    total = sum(totals.values())
    winner = weighted_vote(scores, labels, classes)
    if winner is None or total <= 0:
        return 0.0
    return float(totals[winner] / total)


def vote_predictions(
    frames: dict[str, pd.DataFrame],
    validation_matrix,
    train_matrix,
    k: int,
    step: int = 200,
):
    """Предсказания голосования по соседям для трёх меток одновременно."""
    predicted, _, abstained = vote_all(frames, validation_matrix, train_matrix, (k,), k, step)
    return predicted[k], TAXONOMY_CLASSES, abstained


def vote_support(
    frames: dict[str, pd.DataFrame],
    validation_matrix,
    train_matrix,
    k: int = TOP_K,
    step: int = 200,
) -> dict[str, list[float]]:
    """Доля веса победившей метки среди всех соседей: основа флага «голосование разнесено»."""
    _, support, _ = vote_all(frames, validation_matrix, train_matrix, (k,), k, step)
    return support


def agreement_block(
    validation: pd.DataFrame,
    predicted: dict[str, list[str]],
    support: dict[str, list[float]],
    classes: dict[str, list[str]],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Сравнивает качество там, где победитель собрал половину веса, и там, где нет.

    Если в разнесённом голосовании качество не ниже, флаг «нужно решение
    инженера» ничего не отделяет и его нельзя выдавать за защиту.
    """
    block: dict[str, object] = {
        "rule": f"победа одной метки собрала не меньше {threshold} веса трёх соседей",
        "threshold": threshold,
        "targets": {},
    }
    if not 0.0 <= threshold <= 1.0:
        raise ArchCheckError(f"порог согласия вне отрезка [0, 1]: {threshold}")
    for target in ("service", "root_cause"):
        truth = validation[target].astype(str).to_numpy()
        guess = np.asarray(predicted[target], dtype=str)
        values = np.asarray(support[target], dtype=float)
        if values.size != truth.size:
            raise ArchCheckError(
                f"{target}: {values.size} весов согласия на {truth.size} карточек"
            )
        if guess.size != truth.size:
            raise ArchCheckError(
                f"{target}: {guess.size} предсказаний на {truth.size} карточек"
            )
        solid = values >= threshold
        entry: dict[str, object] = {
            "solid_queries": int(solid.sum()),
            "split_queries": int((~solid).sum()),
            "solid_share": round(float(solid.mean()), 6) if solid.size else 0.0,
        }
        for name, mask in (("solid", solid), ("split", ~solid)):
            if mask.any():
                entry[name] = label_metrics(truth[mask], guess[mask], classes[target])
            else:
                entry[name] = None
        block["targets"][target] = entry  # type: ignore[index]
    return block


# --- метрики --------------------------------------------------------------


def label_metrics(y_true: np.ndarray, y_pred, classes: list[str]) -> dict[str, object]:
    """Точность, F1-macro и recall по полному списку классов таксономии.

    Пустая выдача (`""`) считается ошибкой: она не попадает ни в один класс,
    поэтому доля верных ответов падает, а recall соответствующего класса тоже.
    """
    true = np.asarray(y_true, dtype=str)
    pred = np.asarray(y_pred, dtype=str)
    recall = recall_score(true, pred, labels=list(classes), average=None, zero_division=0)
    payload: dict[str, object] = {
        "accuracy": round(float(accuracy_score(true, pred)), 6),
        "f1_macro": round(
            float(f1_score(true, pred, labels=list(classes), average="macro", zero_division=0)),
            6,
        ),
        "support": {label: int((true == label).sum()) for label in classes},
        "recall": {label: round(float(value), 6) for label, value in zip(classes, recall)},
    }
    if "P1" in classes:
        # При нулевом support recall_score возвращает 0,0, и это читалось бы как
        # «все P1 пропущены», хотя P1 в отрезке просто нет.
        payload["p1_recall"] = payload["recall"]["P1"] if payload["support"]["P1"] else None  # type: ignore[index]
    return payload


def gate_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    classes: list[str],
) -> list[dict[str, object]]:
    """Качество среди ответов, прошедших порог уверенности.

    F1-macro считается по полному списку классов, поэтому порог, отсекающий
    редкий класс, не может улучшить число за счёт выкидывания классов.
    """
    true = np.asarray(y_true, dtype=str)
    pred = np.asarray(y_pred, dtype=str)
    rows: list[dict[str, object]] = []
    for threshold in GATE_THRESHOLDS:
        mask = confidence >= threshold
        covered = int(mask.sum())
        row: dict[str, object] = {
            "threshold": threshold,
            "covered": covered,
            "coverage": round(covered / true.shape[0], 6) if true.shape[0] else 0.0,
        }
        if covered:
            row["accuracy"] = round(float(accuracy_score(true[mask], pred[mask])), 6)
            row["f1_macro"] = round(
                float(
                    f1_score(true[mask], pred[mask], labels=list(classes), average="macro", zero_division=0)
                ),
                6,
            )
        else:
            row["accuracy"] = None
            row["f1_macro"] = None
        if "P1" in classes:
            p1_mask = true == "P1"
            p1_total = int(p1_mask.sum())
            p1_hit = int(((pred == "P1") & mask & p1_mask).sum()) if p1_total else 0
            row["p1_total"] = p1_total
            row["p1_recall"] = round(p1_hit / p1_total, 6) if p1_total else None
        rows.append(row)
    return rows


def classifier_metrics(
    frames: dict[str, pd.DataFrame],
    validation: pd.DataFrame,
    train_matrix,
    validation_matrix,
    classes: dict[str, list[str]],
) -> tuple[dict[str, object], dict[str, LogisticRegression], float]:
    """Три независимых классификатора на одном признаковом пространстве."""
    block: dict[str, object] = {
        "params": dict(CLASSIFIER_PARAMS),
        "targets": {},
        "gate": {},
        "gate_rule": GATE_NOTE,
    }
    models: dict[str, LogisticRegression] = {}
    fit_ms = 0.0

    for target in TARGETS:
        started = time.perf_counter()
        model = LogisticRegression(**CLASSIFIER_PARAMS)
        model.fit(train_matrix, frames["train"][target].astype(str))
        fit_ms += (time.perf_counter() - started) * 1000
        models[target] = model

        probability = model.predict_proba(validation_matrix)
        predicted = model.classes_[probability.argmax(axis=1)]
        truth = validation[target].astype(str).to_numpy()
        metrics = label_metrics(truth, predicted, classes[target])
        labels, counts = np.unique(predicted, return_counts=True)
        metrics["predicted"] = {str(label): int(count) for label, count in zip(labels, counts)}
        block["targets"][target] = metrics  # type: ignore[index]
        block["gate"][target] = gate_curve(truth, predicted, probability.max(axis=1), classes[target])  # type: ignore[index]

    return block, models, fit_ms


# --- вариант B-R: правило критичности ------------------------------------


def severity_table() -> list[tuple[str, str]]:
    """Пары «фраза последствия → критичность» из опубликованной таксономии.

    Таблица `IMPACT_SEVERITY` задаёт критичность для каждого варианта фразы
    последствия, а сами фразы лежат в `SCENARIOS`. Пара собирается по индексу.
    """
    pairs: list[tuple[str, str]] = []
    for scenario in SCENARIOS:
        severities = IMPACT_SEVERITY.get(scenario["id"], {})
        ru_phrases = scenario.get("impacts_ru", ())
        ru_severities = severities.get("ru", ())
        if len(ru_phrases) != len(ru_severities):
            raise ArchCheckError(
                f"{scenario['id']}: {len(ru_phrases)} фраз последствий и "
                f"{len(ru_severities)} оценок критичности"
            )
        for phrase, severity in zip(ru_phrases, ru_severities):
            pairs.append((normalise(phrase), severity))
        impact_en = scenario.get("impact_en")
        if impact_en:
            if not severities.get("en"):
                raise ArchCheckError(f"{scenario['id']}: есть английская фраза без критичности")
            pairs.append((normalise(impact_en), severities["en"]))
    unknown = sorted({severity for _, severity in pairs} - set(SEVERITIES))
    if unknown:
        raise ArchCheckError(f"критичность вне таксономии: {unknown}")
    return sorted(pairs)


def rule_severity(text: str, table: list[tuple[str, str]]) -> str | None:
    """Критичность по самому длинному найденному последствию; None, если фразы нет."""
    lowered = normalise(text)
    best: tuple[str, str] | None = None
    for phrase, severity in table:
        if phrase and phrase in lowered and (best is None or len(phrase) > len(best[0])):
            best = (phrase, severity)
    return best[1] if best else None


def severity_rule_block(
    validation: pd.DataFrame,
    table: list[tuple[str, str]],
    classes: list[str],
) -> tuple[dict[str, object], list[str]]:
    """Покрытие правила и качество на тех карточках, где фраза последствия нашлась."""
    texts = validation[TEXT_COLUMN].astype(str).tolist()
    predicted = [rule_severity(text, table) or "" for text in texts]
    truth = validation["severity"].astype(str).to_numpy()
    covered = np.array([bool(value) for value in predicted])

    block: dict[str, object] = {
        "phrases": len(table),
        "covered": int(covered.sum()),
        "coverage": round(float(covered.mean()), 6) if covered.size else 0.0,
        "accuracy_on_covered": None,
        "f1_macro_on_all": round(
            float(f1_score(truth, np.asarray(predicted), labels=classes, average="macro", zero_division=0)),
            6,
        ),
        "note": (
            "на синтетическом наборе правило воспроизводит определение метки из ЛР 6, "
            "поэтому качество на покрытых карточках измеряет согласованность с определением, "
            "а не перенос на реальные тексты"
        ),
    }
    if covered.any():
        block["accuracy_on_covered"] = round(
            float(accuracy_score(truth[covered], np.asarray(predicted)[covered])),
            6,
        )
    return block, predicted


def hybrid_severity(predicted_rule: list[str], predicted_model: np.ndarray) -> list[str]:
    """Правило первым, модель — там, где фраза последствия не найдена."""
    if len(predicted_rule) != len(predicted_model):
        raise ArchCheckError(
            f"правило дало {len(predicted_rule)} ответов, модель {len(predicted_model)}"
        )
    return [
        rule if rule else str(model)
        for rule, model in zip(predicted_rule, predicted_model)
    ]


# --- задержка -------------------------------------------------------------


def measure_latency(
    frames: dict[str, pd.DataFrame],
    validation: pd.DataFrame,
    vectorizer,
    train_matrix,
    root_model: LogisticRegression,
    table: list[tuple[str, str]],
) -> dict[str, object]:
    """Полный проход принятой конфигурации на один инцидент, один поток.

    Мерится именно тот путь, который описан в архитектуре: услуга — голосование
    по трём соседям, критичность — правило, причина — классификатор. Запасной
    классификатор критичности в замер не входит: на validation правило нашло
    фразу во всех карточках и он ни разу не вызывался.
    """
    labels = {target: frames["train"][target].astype(str).to_numpy() for target in TARGETS}
    durations: list[float] = []
    rule_misses = 0
    for text in validation[TEXT_COLUMN].astype(str):
        started = time.perf_counter()
        cleaned = normalise(text)
        row = vectorizer.transform([cleaned])
        scores = (row @ train_matrix.T).toarray().ravel()
        order = neighbour_order(scores, TOP_K)
        weighted_vote(scores[order], labels["service"][order], TAXONOMY_CLASSES["service"])
        if rule_severity(cleaned, table) is None:
            rule_misses += 1
        root_model.predict_proba(row)
        durations.append((time.perf_counter() - started) * 1000)

    values = np.asarray(durations)
    if values.size == 0:
        return {
            "queries": 0,
            "median_ms": None,
            "mean_ms": None,
            "p95_ms": None,
            "max_ms": None,
            "rule_misses": rule_misses,
            "note": "выборка пуста, замер не выполнен",
        }
    return {
        "queries": int(values.size),
        "median_ms": round(float(np.median(values)), 3),
        "mean_ms": round(float(values.mean()), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "max_ms": round(float(values.max()), 3),
        "rule_misses": rule_misses,
        "note": (
            "принятый путь: нормализация + векторизация + косинус по индексу train + "
            "выборка top-3 + голосование по услуге + правило критичности + predict_proba "
            "классификатора причины; сериализация ответа и запись журнала не входят"
        ),
    }


def index_size(train_matrix) -> dict[str, object]:
    """Размер разреженного индекса, который лежит в памяти сервиса."""
    nbytes = int(train_matrix.data.nbytes + train_matrix.indices.nbytes + train_matrix.indptr.nbytes)
    return {
        "rows": int(train_matrix.shape[0]),
        "features": int(train_matrix.shape[1]),
        "non_zero": int(train_matrix.nnz),
        "bytes": nbytes,
        "megabytes": round(nbytes / 1024 / 1024, 2),
    }


# --- инструменты и внешние ссылки ----------------------------------------


def _licence(info) -> str:
    """Лицензия из метаданных: SPDX-выражение, затем классификатор, затем первая строка."""
    expression = info.get("License-Expression")
    if expression:
        return str(expression)
    classifiers = [item for item in (info.get_all("Classifier") or []) if item.startswith("License ::")]
    if classifiers:
        return classifiers[0].split(" :: ")[-1]
    licence = str(info.get("License") or "")
    first = next((line.strip() for line in licence.splitlines() if line.strip()), "")
    return first[:80] if first else "не указана в метаданных"


def _used_modules() -> dict[str, set[str]]:
    """Модули, которые импортируются кодом сервиса и отдельно — тестами."""
    root = Path(__file__).resolve().parent
    used: dict[str, set[str]] = {"src": set(), "tests": set()}
    for key, folder in (("src", root), ("tests", root.parent / "tests")):
        for path in folder.glob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", text, re.MULTILINE):
                used[key].add(match.group(1))
    return used


def tools_table() -> list[dict[str, object]]:
    """Версия, лицензия и фактический статус каждого средства.

    «Используется» — если модуль импортируется в `src/`, «только в тестах» — если
    он встречается лишь в `tests/`. Установленный пакет, которого никто не
    вызывает, называть используемым нельзя.
    """
    used = _used_modules()
    rows: list[dict[str, object]] = []
    for name, module, role in TOOLS:
        row: dict[str, object] = {"name": name, "role": role}
        if module is None:
            row.update(
                {
                    "version": platform.python_version(),
                    "license": "PSF-2.0",
                    "installed": True,
                    "used_in_src": True,
                    "used_in_tests": True,
                    "status": "используется",
                }
            )
            rows.append(row)
            continue
        try:
            row["version"] = metadata.version(name)
            row["license"] = _licence(metadata.metadata(name))
            row["installed"] = True
            in_src = module in used["src"]
            in_tests = module in used["tests"]
            row["used_in_src"] = in_src
            row["used_in_tests"] = in_tests
            if in_src:
                row["status"] = "используется"
            elif in_tests:
                row["status"] = "только в тестах"
            else:
                row["status"] = "установлен, напрямую не импортируется"
        except metadata.PackageNotFoundError:
            row.update({"version": None, "license": None, "installed": False,
                        "used_in_src": False, "used_in_tests": False,
                        "status": "план: не установлен"})
        except (OSError, ValueError, KeyError):
            # Битый dist-info не должен ронять весь отчёт: помечаем и идём дальше.
            row.update({"version": None, "license": None, "installed": True,
                        "used_in_src": module in used["src"], "used_in_tests": module in used["tests"],
                        "status": "метаданные не читаются"})
        rows.append(row)
    return rows


def references() -> dict[str, object]:
    """Числа предыдущих лабораторных читаются из их файлов, а не переписываются руками."""
    block: dict[str, object] = {}
    if LAB7_METRICS.exists():
        payload = json.loads(LAB7_METRICS.read_text(encoding="utf-8"))
        overall = payload.get("all", {})
        block["lab7"] = {
            "queries": overall.get("queries"),
            "ndcg_at_3_v1": overall.get("ndcg_at_k"),
            "service_hit_at_3": overall.get("service_hit_at_k"),
            "cause_hit_at_3": overall.get("cause_hit_at_k"),
            "both_labels_at_3": overall.get("both_labels_at_k"),
        }
        classification = Path("docs/lab7/results/classification_metrics.json")
        if classification.exists():
            majority = json.loads(classification.read_text(encoding="utf-8"))
            block["lab7"]["majority_f1_macro"] = {
                target: majority[target]["f1_macro"] for target in TARGETS
            }
    if LAB8_SUMMARY.exists():
        payload = json.loads(LAB8_SUMMARY.read_text(encoding="utf-8"))
        block["lab8"] = {
            "ndcg_at_3_v2": payload.get("ndcg", {}).get("v2"),
            "queries": payload.get("queries"),
            "p1_gap": payload.get("p1", {}).get("gap"),
            "alpha": payload.get("thresholds", {}).get("alpha"),
        }
    if LAB9_SUMMARY.exists():
        payload = json.loads(LAB9_SUMMARY.read_text(encoding="utf-8"))
        block["lab9"] = {
            "delta_ndcg": payload.get("delta", {}).get("ndcg@3"),
            "candidate_ndcg": payload.get("candidate", {}).get("ndcg@3"),
            "reranker_accuracy": payload.get("reranker", {}).get("accuracy"),
            "saturated_share": payload.get("reranker", {}).get("saturated_share"),
            "ceiling": payload.get("reproduction", {}).get("ceiling"),
        }
    return block


def reference_checks(cause_accuracy: float) -> dict[str, object]:
    """Сверка с ЛР 9: тот же классификатор причины обязан дать ту же точность."""
    payload: dict[str, object] = {"checked": False}
    if not LAB9_SUMMARY.exists():
        payload["note"] = "сводка ЛР 9 отсутствует, сверка не выполнена"
        return payload
    summary = json.loads(LAB9_SUMMARY.read_text(encoding="utf-8"))
    expected = summary.get("reranker", {}).get("accuracy")
    payload.update(
        {
            "checked": True,
            "lab9_reranker_accuracy": expected,
            "accuracy_now": round(float(cause_accuracy), 6),
            "match": expected is not None and abs(float(expected) - float(cause_accuracy)) < 1e-9,
        }
    )
    return payload


# --- выбор по правилу -----------------------------------------------------


def selection_block(
    vote_per_k: dict[str, dict[str, object]],
    classifier_block: dict[str, object],
    rule_block: dict[str, object],
    hybrid_metrics: dict[str, object],
) -> dict[str, object]:
    """Применяет `SELECTION_RULE` к вариантам в той конфигурации, которая принята.

    Сравниваются только те варианты, которые архитектура действительно может
    взять: голосование по трём карточкам, классификатор, гибрид «правило плюс
    классификатор». Голосование по 10 соседям остаётся в `ceiling` как справка о
    том, сколько стоит объяснимость, а не как участник выбора.
    """
    vote_targets = vote_per_k[str(TOP_K)]["targets"]  # type: ignore[index]
    classifier_targets = classifier_block["targets"]  # type: ignore[index]

    chosen: dict[str, str] = {}
    detail: dict[str, object] = {}
    for target in TARGETS:
        scores = {
            "vote": float(vote_targets[target]["f1_macro"]),  # type: ignore[index]
            "classifier": float(classifier_targets[target]["f1_macro"]),  # type: ignore[index]
        }
        if target == "severity":
            scores["hybrid"] = float(hybrid_metrics["f1_macro"])  # type: ignore[index]
            scores["rule"] = float(rule_block["f1_macro_on_all"])  # type: ignore[index]
        simple_order = ["hybrid", "classifier", "vote", "rule"]
        best_value = max(scores.values())
        contenders = [
            name for name, value in scores.items() if best_value - value <= 0.01 + 1e-12
        ]
        winner = sorted(contenders, key=lambda name: simple_order.index(name))[0]
        runner_up = max((value for name, value in scores.items() if name != winner), default=0.0)
        chosen[target] = winner
        detail[target] = {
            "f1_macro": {name: round(value, 6) for name, value in scores.items()},
            "chosen": winner,
            "lead_over_runner_up": round(scores[winner] - runner_up, 6),
        }

    ceiling: dict[str, object] = {}
    for k in VOTE_KS:
        means = [
            float(vote_per_k[str(k)]["targets"][target]["f1_macro"])  # type: ignore[index]
            for target in TARGETS
        ]
        ceiling[str(k)] = {
            "targets": {
                target: float(vote_per_k[str(k)]["targets"][target]["f1_macro"])  # type: ignore[index]
                for target in TARGETS
            },
            "mean_f1_macro": round(float(np.mean(means)), 6),
        }
    best_k = max(VOTE_KS, key=lambda k: ceiling[str(k)]["mean_f1_macro"])  # type: ignore[index]
    cost = {
        target: round(
            float(ceiling[str(best_k)]["targets"][target])  # type: ignore[index]
            - float(ceiling[str(TOP_K)]["targets"][target]),  # type: ignore[index]
            6,
        )
        for target in TARGETS
    }

    decision: dict[str, object] = {
        "severity": {
            "method": chosen["severity"],
            "note": "правило ставит метку, классификатор включается только там, где фразы последствия нет",
        },
        "service": {
            "method": chosen["service"],
            "k": TOP_K,
            "reason": "метка объясняется теми же карточками, которые инженер видит в ответе",
        },
        "root_cause": {"method": chosen["root_cause"]},
    }

    return {
        "rule": SELECTION_RULE,
        "chosen": chosen,
        "decision": decision,
        "detail": detail,
        "ceiling": {
            "best_k": best_k,
            "by_k": ceiling,
            "cost_of_showing_top3": cost,
            "note": "потолок по F1-macro; в выборе не участвует, потому что метку нельзя объяснить невидимыми карточками",
        },
    }


# --- сборка отчёта --------------------------------------------------------


def gates_block(
    selection: dict[str, object],
    hybrid_metrics: dict[str, object],
    latency: dict[str, object],
) -> dict[str, object]:
    """Проверяет ворота проекта: цели ЛР 1 и бюджет задержки.

    Берётся конфигурация, принятая в `selection.chosen`, а не лучшая из всех
    измеренных: ворота проверяют то, что пойдёт в alpha.
    """

    def scores_of(target: str) -> dict[str, object] | None:
        block = selection.get("detail")  # type: ignore[union-attr]
        if not isinstance(block, dict):
            return None
        entry = block.get(target)
        if not isinstance(entry, dict):
            return None
        scores = entry.get("f1_macro")
        return scores if isinstance(scores, dict) else None

    chosen = selection.get("chosen")  # type: ignore[union-attr]
    if not isinstance(chosen, dict):
        raise ArchCheckError("для ворот нужен выбор с полями detail и chosen")

    adopted_scores: dict[str, float] = {}
    missing: list[str] = []
    for target in TARGETS:
        scores = scores_of(target)
        if scores is None or chosen.get(target) not in scores:
            missing.append(target)
            continue
        adopted_scores[target] = float(scores[chosen[target]])
    if missing:
        raise ArchCheckError(f"в выборе нет принятого варианта для меток: {missing}")

    p1_recall = hybrid_metrics.get("p1_recall")
    p95 = latency.get("p95_ms")
    latency_met = p95 is not None and float(p95) <= LATENCY_BUDGET_MS
    return {
        "plan_targets": dict(PLAN_TARGETS),
        "adopted_f1_macro": {target: round(value, 6) for target, value in adopted_scores.items()},
        "targets_met": {
            target: adopted_scores[target] >= PLAN_TARGETS[target] for target in TARGETS
        },
        "p1_recall": p1_recall,
        "p1_recall_met": bool(p1_recall is not None and float(p1_recall) >= PLAN_TARGETS["p1_recall"]),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "latency_p95_ms": p95,
        "latency_met": latency_met,
        "note": (
            "критичность измеряется по правилу из ЛР 6, то есть по определению метки; "
            "выполнение цели на синтетике не доказывает качество на реальных карточках"
        ),
    }


def build_report() -> dict[str, object]:
    """Считает все блоки проверки и возвращает словарь для JSON."""
    frames = load_selected_splits(list(SPLITS_USED))
    validation = frames["validation"]
    classes = TAXONOMY_CLASSES
    missing_in_train = {
        target: sorted(set(classes[target]) - set(frames["train"][target].astype(str)))
        for target in TARGETS
    }
    data_versions = sorted(set(frames["train"]["dataset_version"].astype(str)))

    started = time.perf_counter()
    vectorizer, train_matrix, validation_matrix = fit_space(frames)
    index_build_ms = (time.perf_counter() - started) * 1000

    manifest = manifest_check(vectorizer)
    if not manifest.get("checked"):
        raise ArchCheckError(
            "нет манифеста словаря ЛР 6, сверять не с чем: " + json.dumps(manifest, ensure_ascii=False)
        )
    if not manifest.get("match"):
        raise ArchCheckError(
            "словарь не совпал с манифестом ЛР 6: " + json.dumps(manifest, ensure_ascii=False)
        )

    predictions_by_k, support, abstained = vote_all(frames, validation_matrix, train_matrix)
    vote_block: dict[str, object] = {
        "rule": "метка = сумма косинусов соседей; ничья — ближайший сосед, затем алфавит",
        "per_k": {},
    }
    for k in VOTE_KS:
        predicted = predictions_by_k[k]
        vote_block["per_k"][str(k)] = {  # type: ignore[index]
            "abstained": abstained,
            "targets": {
                target: label_metrics(validation[target].astype(str).to_numpy(), predicted[target], classes[target])
                for target in TARGETS
            },
        }

    agreement = agreement_block(validation, predictions_by_k[TOP_K], support, classes)

    classifier_block, models, fit_ms = classifier_metrics(
        frames, validation, train_matrix, validation_matrix, classes
    )

    table = severity_table()
    rule_block, predicted_rule = severity_rule_block(validation, table, classes["severity"])
    severity_model = models["severity"]
    probability = severity_model.predict_proba(validation_matrix)
    predicted_model = severity_model.classes_[probability.argmax(axis=1)]
    hybrid_metrics = label_metrics(
        validation["severity"].astype(str).to_numpy(),
        hybrid_severity(predicted_rule, predicted_model),
        classes["severity"],
    )

    latency = measure_latency(
        frames, validation, vectorizer, train_matrix, models["root_cause"], table
    )
    latency["index_build_ms"] = round(index_build_ms, 2)
    latency["model_fit_ms"] = round(fit_ms, 2)

    vote_per_k = vote_block["per_k"]  # type: ignore[index]
    selection = selection_block(vote_per_k, classifier_block, rule_block, hybrid_metrics)
    return {
        "version": CHECK_VERSION,
        "data_version": ", ".join(data_versions),
        "splits_used": list(SPLITS_USED),
        "test_used": "test" in SPLITS_USED,
        "selection_split": "validation",
        "queries": int(validation.shape[0]),
        "train_cards": int(frames["train"].shape[0]),
        "classes": {
            "taxonomy": classes,
            "missing_in_train": missing_in_train,
            "note": "классы берутся из таксономии, поэтому F1-macro не зависит от состава train",
        },
        "contract": {
            "endpoint": "POST /predict",
            "request_fields": ["title", "description", "monitoring_source", "region"],
            "data_field_map": {"monitoring_source": "source_system"},
            "min_tokens": MIN_TOKEN_COUNT,
            "max_chars": 4000,
            "note": "имена полей взяты из README, а не придуманы в схеме",
        },
        "features": {
            "params": {key: list(value) if isinstance(value, tuple) else value for key, value in VECTORIZER_PARAMS.items()},
            "vocabulary_size": len(vectorizer.vocabulary_),
            "vocabulary_sha256": vocabulary_hash(vectorizer),
            "manifest": manifest,
        },
        "variants": {
            "vote": vote_block,
            "classifiers": classifier_block,
            "severity_rule": rule_block,
            "severity_hybrid": hybrid_metrics,
            "vote_agreement": agreement,
        },
        "latency": latency,
        "index": index_size(train_matrix),
        "references": references(),
        "reference_checks": reference_checks(
            classifier_block["targets"]["root_cause"]["accuracy"]  # type: ignore[index]
        ),
        "selection": selection,
        "gates": gates_block(selection, hybrid_metrics, latency),
        "environment": environment(),
        "tools": tools_table(),
        "input_hashes": input_hashes(list(SPLITS_USED)),
    }


def run(results_dir: Path = RESULTS_DIR) -> int:
    """Пишет `check.json` с измерениями; test не читается.

    Отчёт не записывается, если сверка с ЛР 9 не подтвердилась или её не с чем
    сделать: тогда документы, где написано «совпало до шестого знака», стали бы
    неправдой.
    """
    try:
        report = build_report()
    except (FileNotFoundError, ArchCheckError, json.JSONDecodeError) as error:
        print(f"Ошибка: {error}")
        return 1

    reference = report["reference_checks"]
    if not reference.get("checked") or not reference.get("match"):
        print("Ошибка: сверка с ЛР 9 не подтвердилась, отчёт не записан: "
              + json.dumps(reference, ensure_ascii=False))
        return 1

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "check.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"отчёт: {path}")
    print(f"выбор: {json.dumps(report['selection']['chosen'], ensure_ascii=False)}")  # type: ignore[index]
    print(f"задержка: медиана {report['latency']['median_ms']} мс, p95 {report['latency']['p95_ms']} мс")  # type: ignore[index]
    return 0


# --- схема компонентов ----------------------------------------------------


BOX_FILL = "#eef2f7"
BOX_EDGE = "#2f4b6e"
HUMAN_FILL = "#fdf1d6"
SPARE_FILL = "#fafafa"
TEXT_COLOR = "#1c2733"

MAIN_BOXES = (
    ("1  Вход", "POST /predict\nJSON: title, description,\nmonitoring_source, region"),
    ("2  Валидация", "Pydantic: обязательные поля,\nне меньше 5 слов, не больше\n4000 символов; 400 / 413 / 422"),
    ("3  Модель и правила", "TF-IDF (словарь train):\nкритичность — правило,\nуслуга — голосование,\nпричина — классификатор"),
    ("4  Слой знаний", "индекс train-карточек,\nтаксономия 7 услуг / 6 причин,\nfeatures_manifest.json,\nrunbook — заглушка"),
    ("5  Поиск top-3", "косинус по индексу train,\nфильтр rights_status =\ndemo_allowed"),
    ("6  Фильтрация", "нет слов в словаре, нет\nкарточек, разнесённое\nголосование → «нужно\nрешение инженера»"),
    ("7  Выдача", "JSON: severity, service,\nroot_cause, similar[3],\nstatus, model_version"),
    ("8  Инженер", "подтверждение или правка;\nрешение остаётся\nза человеком"),
    ("9  Журнал", "runtime/feedback.jsonl:\nrequest_id, action, actor,\nproposed_*; без переобучения"),
)

SPARE_BOXES = (
    ("2  Валидация → 400 / 413 / 422", "пустой, слишком короткий или\nдлинный текст, неверный\nContent-Type: модель не\nвызывается"),
    ("3  Модель → fallback / 503", "нет файла модели или словарь\nне сошёлся с манифестом:\nтолько top-3, status=fallback;\nнет индекса — 503"),
    ("5  Поиск → no_matches", "нулевой вектор или всё\nотфильтровано: similar=[],\nметки не выдаются,\nstatus=no_matches"),
)


def _box(ax, x, y, width, height, title, body, fill, dashed=False):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=BOX_EDGE,
            facecolor=fill,
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(x + width / 2, y + height - 0.16, title, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=BOX_EDGE)
    ax.text(x + width / 2, y + height - 0.55, body, ha="center", va="top",
            fontsize=7.8, color=TEXT_COLOR, linespacing=1.45)


def _arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=BOX_EDGE, linewidth=1.2, shrinkA=0, shrinkB=0),
    )


def draw_scheme(path: Path = FIGURE_PATH) -> int:
    """Рисует схему компонентов: основной путь, решение человека, резервные пути."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    figure, axis = plt.subplots(figsize=(13.4, 8.2))
    axis.set_xlim(0, 13.4)
    axis.set_ylim(0, 8.6)
    axis.axis("off")

    width, height = 2.4, 1.75
    row_one, row_two = 5.95, 3.45
    columns = [0.35, 2.95, 5.55, 8.15, 10.75]

    for index in range(5):
        title, body = MAIN_BOXES[index]
        _box(axis, columns[index], row_one, width, height, title, body, BOX_FILL)

    # Второй ряд идёт справа налево: поток заворачивается под правым краем.
    for position, index in enumerate(range(5, 9)):
        title, body = MAIN_BOXES[index]
        x = columns[4 - position]
        fill = HUMAN_FILL if title.startswith("8") else BOX_FILL
        _box(axis, x, row_two, width, height, title, body, fill)

    for index in range(4):
        _arrow(axis, (columns[index] + width, row_one + height / 2), (columns[index + 1], row_one + height / 2))
    _arrow(axis, (columns[4] + width / 2, row_one), (columns[4] + width / 2, row_two + height))
    for position in range(3):
        _arrow(
            axis,
            (columns[4 - position], row_two + height / 2),
            (columns[3 - position] + width, row_two + height / 2),
        )

    axis.text(0.35, 8.42, "SRE Triage Assistant: маршрут одного инцидента в alpha-версии",
              fontsize=12.5, fontweight="bold", color=TEXT_COLOR, ha="left")
    axis.text(0.35, 8.05, "все шаги локальные; ни один шаг не обращается к внешнему платному сервису",
              fontsize=8.5, color="#55606b", ha="left")

    from matplotlib.patches import FancyBboxPatch

    axis.add_patch(
        FancyBboxPatch((0.2, 0.45), 13.0, 2.35, boxstyle="round,pad=0.05,rounding_size=0.12",
                       linewidth=1.0, edgecolor="#8a949e", facecolor=SPARE_FILL, linestyle="--")
    )
    axis.text(0.35, 2.62, "Резервные пути и ошибки — отдельно от основного маршрута",
              fontsize=9.5, fontweight="bold", color="#55606b", ha="left")
    spare_width = 4.0
    for position, (title, body) in enumerate(SPARE_BOXES):
        _box(axis, 0.4 + position * 4.35, 0.7, spare_width, 1.6, title, body, SPARE_FILL, dashed=True)

    axis.text(0.35, 0.18, "оранжевый — решение человека, пунктир — резервный путь; данные между шагами: строка запроса, идентификатор и оценка карточки, поля ответа, событие обратной связи",
              fontsize=7.2, color="#55606b", ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"схема: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="измерить варианты на validation")
    run_parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)

    figure_parser = subparsers.add_parser("figure", help="нарисовать схему компонентов")
    figure_parser.add_argument("--path", type=Path, default=FIGURE_PATH)

    args = parser.parse_args()
    if args.command == "run":
        return run(Path(args.results_dir))
    return draw_scheme(Path(args.path))


if __name__ == "__main__":
    raise SystemExit(main())
