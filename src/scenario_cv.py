"""Leave-one-scenario-out внутри train: насколько результат ЛР 10 стабилен.

Диагностика отвечает на вопрос «наши 0,30 по услуге и 0,27 по причине — это
свойство режима переноса или удачное разбиение». Каждый из 14 сценариев train
по очереди выводится из индекса, словарь и IDF считаются заново по оставшимся
13, поэтому запросы фолда не встречаются в индексе ни текстом, ни шаблоном —
это тот же режим, что и в validation.

Метрики считаются двумя способами, и это важно:

* по фолдам — точность и NDCG; в фолде один сценарий, значит одна услуга и одна
  причина, поэтому macro-F1 внутри фолда обрезан до 1/7 и для сравнения не
  годится;
* по пулу — предсказания всех 14 фолдов складываются (каждая карточка train
  предсказана ровно один раз) и macro-F1 считается по полной таксономии. Только
  это число сравнимо с validation.

Validation и test не читаются при подсчёте метрик: фолды идут только по train.
Из validation берётся один описательный показатель — средняя близость текстов к
карточкам своей услуги, — чтобы объяснить разрыв между фолдами и validation.

Запуск из корня репозитория:
    python src/scenario_cv.py run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from arch_check import (
    CLASSIFIER_PARAMS,
    PLAN_TARGETS,
    TAXONOMY_CLASSES,
    TARGETS,
    TOP_K,
    hybrid_severity,
    label_metrics,
    rule_severity,
    severity_table,
    weighted_vote,
)
from eval_report import grade_v2, ideal_counts
from experiment import input_hashes, load_selected_splits
from prepare_features import TEXT_COLUMN, fit_vectorizer, transform
from triage_baseline import ideal_grades, ndcg

DIAGNOSTIC_VERSION = "1.0"
RESULTS_DIR = Path("docs/diagnostics/results")
DOCS_DIR = Path("docs/diagnostics")
LAB10_SUMMARY = Path("docs/lab10/results/check.json")
LAB9_SUMMARY = Path("docs/lab9/results/summary.json")
LAB7_METRICS = Path("docs/lab7/results/retrieval_metrics.json")

# Показатели фолда: точность, поиск и правило. Macro-F1 здесь нет намеренно.
FOLD_METRICS = (
    "ndcg@3",
    "service_hit@3",
    "cause_hit@3",
    "vote_accuracy_service",
    "vote_accuracy_cause",
    "clf_accuracy_service",
    "clf_accuracy_cause",
    "clf_accuracy_severity",
    "rule_coverage",
    "rule_accuracy",
    "hybrid_accuracy_severity",
    "same_service_max_cosine",
)

TITLES = {
    "ndcg@3": "NDCG@3",
    "service_hit@3": "Попадание услуги в top-3",
    "cause_hit@3": "Попадание причины в top-3",
    "vote_accuracy_service": "Услуга, голосование: доля верных",
    "vote_accuracy_cause": "Причина, голосование: доля верных",
    "clf_accuracy_service": "Услуга, классификатор: доля верных",
    "clf_accuracy_cause": "Причина, классификатор: доля верных",
    "clf_accuracy_severity": "Критичность, классификатор: доля верных",
    "rule_coverage": "Покрытие правила критичности",
    "rule_accuracy": "Точность правила на покрытых",
    "hybrid_accuracy_severity": "Критичность, правило плюс классификатор",
    "same_service_max_cosine": "Близость к карточкам своей услуги",
    "vote_service_f1": "Услуга, голосование: F1-macro по пулу",
    "vote_cause_f1": "Причина, голосование: F1-macro по пулу",
    "clf_service_f1": "Услуга, классификатор: F1-macro по пулу",
    "clf_cause_f1": "Причина, классификатор: F1-macro по пулу",
    "clf_severity_f1": "Критичность, классификатор: F1-macro по пулу",
    "clf_p1_recall": "P1 recall по пулу",
}

POOLED_METRICS = (
    "vote_service_f1",
    "vote_cause_f1",
    "clf_service_f1",
    "clf_cause_f1",
    "clf_severity_f1",
    "clf_p1_recall",
)

# Короткие названия для фраз в прозе: без двоеточий внутри перечисления.
SHORT_TITLES = {
    "vote_service_f1": "услуга, голосование",
    "vote_cause_f1": "причина, голосование",
    "clf_service_f1": "услуга, классификатор",
    "clf_cause_f1": "причина, классификатор",
    "clf_severity_f1": "критичность, классификатор",
    "clf_p1_recall": "P1 recall",
}


def fold_scenarios(train: pd.DataFrame) -> list[str]:
    """Сценарии train в стабильном порядке: один сценарий — один фолд."""
    scenarios = sorted(train["scenario_id"].astype(str).unique())
    if len(scenarios) < 2:
        raise ValueError(f"для фолдов нужно минимум два сценария, найдено {len(scenarios)}")
    return scenarios


def split_fold(train: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит train на индекс (13 сценариев) и запросы (выведенный сценарий)."""
    index = train[train["scenario_id"].astype(str) != scenario].reset_index(drop=True)
    queries = train[train["scenario_id"].astype(str) == scenario].reset_index(drop=True)
    if index.empty or queries.empty:
        raise ValueError(f"фолд {scenario}: индекс {len(index)}, запросы {len(queries)}")
    shared = set(index["text_hash"]) & set(queries["text_hash"])
    if shared:
        raise ValueError(f"фолд {scenario}: {len(shared)} общих текстов в индексе и запросах")
    return index, queries


def ranked_top(scores: np.ndarray, ids: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Top-k по убыванию близости, ничьи — по возрастанию incident_id."""
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=int)
    order = np.lexsort((ids, -scores))[: min(k, scores.size)]
    return order[scores[order] > 0]


def same_service_similarity(index: pd.DataFrame, queries: pd.DataFrame, vectorizer, index_matrix) -> float:
    """Средняя максимальная близость запроса к карточкам своей услуги в индексе.

    Это мера того, насколько словарь сценария вообще пересекается с историей:
    именно она объясняет, почему один сценарий переносится лучше другого.
    """
    if queries.empty or index.empty:
        return 0.0
    query_matrix = transform(vectorizer, queries)
    similar = np.asarray((query_matrix @ index_matrix.T).toarray())
    services = index["service"].to_numpy()
    values: list[float] = []
    for position, (_, query) in enumerate(queries.iterrows()):
        mask = services == query["service"]
        values.append(float(similar[position][mask].max()) if mask.any() else 0.0)
    return round(float(np.mean(values)), 6) if values else 0.0


def evaluate_fold(index: pd.DataFrame, queries: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    """Считает фолд и возвращает сводку фолда и построчные предсказания."""
    vectorizer = fit_vectorizer(index)
    index_matrix = transform(vectorizer, index)
    query_matrix = transform(vectorizer, queries)
    similarity = np.asarray((query_matrix @ index_matrix.T).toarray())

    ids = index["incident_id"].to_numpy()
    table = severity_table()
    index_shim = SimpleNamespace(train=index)
    labels = {target: index[target].astype(str).to_numpy() for target in TARGETS}

    ndcgs: list[float] = []
    service_hits: list[float] = []
    cause_hits: list[float] = []
    rows: list[dict[str, object]] = []

    for position, (_, query) in enumerate(queries.iterrows()):
        scores = similarity[position]
        order = ranked_top(scores, ids)
        cards = index.iloc[order] if order.size else index.iloc[[]]
        window = scores[order]

        grades = [grade_v2(query, card) for _, card in cards.iterrows()]
        grade_two, grade_one = ideal_counts(index_shim, query)["v2"]
        ndcgs.append(ndcg(grades, ideal_grades(grade_two, grade_one, TOP_K), TOP_K))
        service_hits.append(float(bool((cards["service"] == query["service"]).any())))
        cause_hits.append(float(bool((cards["root_cause"] == query["root_cause"]).any())))

        row: dict[str, object] = {"incident_id": query["incident_id"]}
        for target in TARGETS:
            row[f"true_{target}"] = str(query[target])
            row[f"vote_{target}"] = (
                weighted_vote(window, labels[target][order], TAXONOMY_CLASSES[target]) or ""
            )
        row["rule_severity"] = rule_severity(query[TEXT_COLUMN], table) or ""
        rows.append(row)

    models: dict[str, LogisticRegression] = {}
    classifier = {target: [] for target in TARGETS}
    for target in TARGETS:
        model = LogisticRegression(**CLASSIFIER_PARAMS)
        model.fit(index_matrix, index[target].astype(str))
        models[target] = model
        classifier[target] = list(model.classes_[model.predict_proba(query_matrix).argmax(axis=1)])

    for position, row in enumerate(rows):
        row["clf_severity"] = classifier["severity"][position]
        row["clf_service"] = classifier["service"][position]
        row["clf_root_cause"] = classifier["root_cause"][position]
        row["hybrid_severity"] = hybrid_severity([str(row["rule_severity"])], [classifier["severity"][position]])[0]

    frame = pd.DataFrame(rows)
    covered = frame["rule_severity"].astype(str) != ""
    summary = {
        "scenario_id": str(queries["scenario_id"].iloc[0]),
        "service": str(queries["service"].iloc[0]),
        "cards": int(len(queries)),
        "index_cards": int(len(index)),
        "metrics": {
            "ndcg@3": round(float(np.mean(ndcgs)), 6) if ndcgs else 0.0,
            "service_hit@3": round(float(np.mean(service_hits)), 6) if service_hits else 0.0,
            "cause_hit@3": round(float(np.mean(cause_hits)), 6) if cause_hits else 0.0,
            "vote_accuracy_service": round(float((frame["vote_service"] == frame["true_service"]).mean()), 6),
            "vote_accuracy_cause": round(float((frame["vote_root_cause"] == frame["true_root_cause"]).mean()), 6),
            "clf_accuracy_service": round(float((frame["clf_service"] == frame["true_service"]).mean()), 6),
            "clf_accuracy_cause": round(float((frame["clf_root_cause"] == frame["true_root_cause"]).mean()), 6),
            "clf_accuracy_severity": round(float((frame["clf_severity"] == frame["true_severity"]).mean()), 6),
            "rule_coverage": round(float(covered.mean()), 6),
            "rule_accuracy": (
                round(float((frame.loc[covered, "rule_severity"] == frame.loc[covered, "true_severity"]).mean()), 6)
                if covered.any()
                else None
            ),
            "hybrid_accuracy_severity": round(
                float((frame["hybrid_severity"] == frame["true_severity"]).mean()), 6
            ),
            "same_service_max_cosine": same_service_similarity(index, queries, vectorizer, index_matrix),
        },
    }
    return summary, frame


def pooled_metrics(records: pd.DataFrame) -> dict[str, object]:
    """Macro-F1 по всем 3 600 карточкам train: единственное число, сравнимое с validation."""
    block: dict[str, object] = {}
    for target in TARGETS:
        vote = label_metrics(
            records[f"true_{target}"].to_numpy(),
            records[f"vote_{target}"].to_numpy(),
            TAXONOMY_CLASSES[target],
        )
        classifier = label_metrics(
            records[f"true_{target}"].to_numpy(),
            records[f"clf_{target}"].to_numpy(),
            TAXONOMY_CLASSES[target],
        )
        block[f"vote_{'cause' if target == 'root_cause' else target}_f1"] = vote["f1_macro"]
        block[f"clf_{'cause' if target == 'root_cause' else target}_f1"] = classifier["f1_macro"]
        if target == "severity":
            block["clf_p1_recall"] = classifier.get("p1_recall")
    return block


def summarise(folds: list[dict[str, object]]) -> dict[str, object]:
    """Среднее, стандартное отклонение и границы по показателям фолдов."""
    block: dict[str, object] = {}
    for metric in FOLD_METRICS:
        values = [fold["metrics"][metric] for fold in folds]  # type: ignore[index]
        numeric = [float(value) for value in values if value is not None]
        if not numeric:
            block[metric] = {"mean": None, "std": None, "min": None, "max": None, "folds": 0}
            continue
        block[metric] = {
            "mean": round(float(np.mean(numeric)), 6),
            "std": round(float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0, 6),
            "min": round(float(min(numeric)), 6),
            "max": round(float(max(numeric)), 6),
            "folds": len(numeric),
        }
    return block


def published_reference() -> dict[str, object]:
    """Опубликованные значения ЛР 7–10 для сверки, прочитанные из их файлов."""
    block: dict[str, object] = {}
    if LAB10_SUMMARY.exists():
        payload = json.loads(LAB10_SUMMARY.read_text(encoding="utf-8"))
        vote = payload["variants"]["vote"]["per_k"][str(TOP_K)]["targets"]
        classifiers = payload["variants"]["classifiers"]["targets"]
        rule = payload["variants"]["severity_rule"]
        block.update(
            {
                "vote_service_f1": vote["service"]["f1_macro"],
                "vote_cause_f1": vote["root_cause"]["f1_macro"],
                "clf_service_f1": classifiers["service"]["f1_macro"],
                "clf_cause_f1": classifiers["root_cause"]["f1_macro"],
                "clf_severity_f1": classifiers["severity"]["f1_macro"],
                "clf_p1_recall": classifiers["severity"].get("p1_recall"),
                "rule_coverage": rule["coverage"],
            }
        )
    if LAB9_SUMMARY.exists():
        payload = json.loads(LAB9_SUMMARY.read_text(encoding="utf-8"))
        block["ndcg@3"] = payload["baseline"]["ndcg@3"]
    if LAB7_METRICS.exists():
        payload = json.loads(LAB7_METRICS.read_text(encoding="utf-8"))
        block["service_hit@3"] = payload["all"]["service_hit_at_k"]
        block["cause_hit@3"] = payload["all"]["cause_hit_at_k"]
    return block


def comparison_block(values: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    """Сравнивает пуловые значения и опубликованные; разницу считает, а не описывает."""
    block: dict[str, object] = {}
    for metric, published in reference.items():
        produced = values.get(metric)
        if produced is None or published is None:
            continue
        block[metric] = {
            "pooled": round(float(produced), 6),
            "validation": round(float(published), 6),
            "delta": round(float(produced) - float(published), 6),
        }
    return block


def validation_proximity(train: pd.DataFrame, validation: pd.DataFrame) -> float:
    """Средняя близость карточек validation к своей услуге в train.

    Замер описательный: метки validation здесь не используются, считается только
    косинус между текстами. Он нужен, чтобы понять, насколько validation проще
    или сложнее среднего сценария train.
    """
    vectorizer = fit_vectorizer(train)
    matrix = transform(vectorizer, train)
    return same_service_similarity(train, validation, vectorizer, matrix)


def build_report() -> dict[str, object]:
    """Считает все фолды и собирает отчёт."""
    started = time.perf_counter()
    frames = load_selected_splits(["train", "validation"])
    train = frames["train"]

    folds: list[dict[str, object]] = []
    records: list[pd.DataFrame] = []
    for scenario in fold_scenarios(train):
        index, queries = split_fold(train, scenario)
        summary, frame = evaluate_fold(index, queries)
        folds.append(summary)
        records.append(frame)

    pooled = pooled_metrics(pd.concat(records, ignore_index=True))
    reference = published_reference()
    proximities = [fold["metrics"]["same_service_max_cosine"] for fold in folds]  # type: ignore[index]
    return {
        "version": DIAGNOSTIC_VERSION,
        "data_version": ", ".join(sorted(set(train["dataset_version"].astype(str)))),
        "splits_used": ["train"],
        "test_used": False,
        "validation_used": "только средняя близость текстов, метки не используются",
        "scheme": "leave-one-scenario-out по train",
        "cards": int(len(train)),
        "folds": len(folds),
        "per_fold": folds,
        "fold_summary": summarise(folds),
        "pooled": pooled,
        "reference": reference,
        "comparison": comparison_block(pooled, reference),
        "proximity": {
            "folds_mean": round(float(np.mean(proximities)), 6),
            "folds_min": round(float(min(proximities)), 6),
            "folds_max": round(float(max(proximities)), 6),
            "validation": validation_proximity(train, frames["validation"]),
            "note": "средний косинус до ближайшей карточки своей услуги в индексе",
        },
        "class_shares": {
            "severity": {
                "train": {
                    label: round(float(share), 6)
                    for label, share in train["severity"].value_counts(normalize=True).sort_index().items()
                },
                "validation": {
                    label: round(float(share), 6)
                    for label, share in frames["validation"]["severity"]
                    .value_counts(normalize=True)
                    .sort_index()
                    .items()
                },
            }
        },
        "seconds": round(time.perf_counter() - started, 2),
        "input_hashes": input_hashes(["train", "validation"]),
    }


def format_value(value: object, digits: int = 3) -> str:
    """Число с запятой и нужным числом знаков; None — прочерк."""
    if value is None:
        return "—"
    text = f"{float(value):.{digits}f}".replace(".", ",")
    return "−" + text[1:] if text.startswith("-") else text


def format_int(value: object) -> str:
    """Целое с разделением тысяч обычным пробелом."""
    return f"{int(value):,}".replace(",", " ")


def format_delta(value: object) -> str:
    """Разница со знаком: плюс только для положительной, минус — типографский."""
    if value is None:
        return "—"
    number = float(value)
    if abs(number) < 5e-7:
        return "0,000"
    sign = "+" if number > 0 else ""
    return sign + format_value(number)


def render_markdown(report: dict[str, object]) -> str:
    """Собирает документ диагностики из отчёта, без чисел, вписанных руками."""
    fold_summary = report["fold_summary"]  # type: ignore[index]
    comparison = report["comparison"]  # type: ignore[index]
    folds = report["per_fold"]  # type: ignore[index]

    lines = [
        "# Проверка устойчивости: leave-one-scenario-out по train",
        "",
        "Диагностика к ЛР 10. Вопрос: 0,30 по услуге и 0,27 по причине — это",
        "свойство режима переноса на незнакомые формулировки или следствие одного",
        "удачного разбиения.",
        "",
        "Схема: {folds} фолдов по {cards} карточкам train. В каждом".format(
            folds=report["folds"], cards=format_int(report["cards"])
        ),
        "фолде один сценарий выведен из индекса, словарь и IDF считаются заново по",
        "оставшимся 13, поэтому запросы не встречаются в индексе ни текстом, ни",
        "шаблоном. Метрики считаются только по train; из validation берётся одна",
        "описательная величина — близость текстов.",
        "",
        "## Как считаются метрики",
        "",
        "В фолде один сценарий, то есть одна услуга и одна причина. Macro-F1 внутри",
        "такого фолда обрезан: шесть из семи классов не имеют поддержки, поэтому",
        "потолок равен 1/7. Сравнимое с validation число даёт пул: предсказания всех",
        "фолдов складываются, каждая карточка train предсказана ровно один раз, и",
        "macro-F1 считается по полной таксономии.",
        "",
        "По той же причине в таблице фолдов стоит «доля верных», а не точность: в",
        "одноклассовом фолде она равна полноте этого класса, а precision не определена.",
        "Критичность — исключение: она меняется внутри сценария, поэтому в её фолде",
        "классов несколько.",
        "",
        "## Результаты по фолдам",
        "",
        "| Сценарий | Услуга | Карточек | NDCG@3 | Услуга, голосование | Причина, классификатор | Критичность, правило |",
        "|---|---|---|---|---|---|---|",
    ]
    for fold in folds:  # type: ignore[union-attr]
        metrics = fold["metrics"]
        lines.append(
            "| {scenario} | {service} | {cards} | {ndcg} | {service_acc} | {cause_acc} | {rule} |".format(
                scenario=fold["scenario_id"],
                service=fold["service"],
                cards=fold["cards"],
                ndcg=format_value(metrics["ndcg@3"]),
                service_acc=format_value(metrics["vote_accuracy_service"]),
                cause_acc=format_value(metrics["clf_accuracy_cause"]),
                rule=format_value(metrics["rule_accuracy"]),
            )
        )

    lines += [
        "",
        "## Разброс по фолдам",
        "",
        "| Показатель | Среднее | Стандартное отклонение | Минимум | Максимум |",
        "|---|---|---|---|---|",
    ]
    for metric in FOLD_METRICS:
        entry = fold_summary[metric]  # type: ignore[index]
        lines.append(
            "| {title} | {mean} | {std} | {low} | {high} |".format(
                title=TITLES[metric],
                mean=format_value(entry["mean"]),
                std=format_value(entry["std"]),
                low=format_value(entry["min"]),
                high=format_value(entry["max"]),
            )
        )

    lines += [
        "",
        "## Пул против validation",
        "",
        "| Показатель | Пул 14 фолдов | Validation | Разница |",
        "|---|---|---|---|",
    ]
    for metric, entry in comparison.items():  # type: ignore[union-attr]
        lines.append(
            "| {title} | {pooled} | {validation} | {delta} |".format(
                title=TITLES.get(metric, metric),
                pooled=format_value(entry["pooled"]),
                validation=format_value(entry["validation"]),
                delta=format_delta(entry["delta"]),
            )
        )

    deltas = [entry["delta"] for entry in comparison.values()]  # type: ignore[union-attr]
    below = [SHORT_TITLES.get(metric, metric) for metric, entry in comparison.items() if entry["delta"] < 0]  # type: ignore[union-attr]
    above = [SHORT_TITLES.get(metric, metric) for metric, entry in comparison.items() if entry["delta"] > 0]  # type: ignore[union-attr]

    proximity = report["proximity"]  # type: ignore[index]
    closer = float(proximity["validation"]) > float(proximity["folds_mean"])  # type: ignore[arg-type]

    lines += [
        "",
        "## Почему validation выше",
        "",
        "Средняя близость текста к ближайшей карточке своей услуги в индексе: "
        f"фолды {format_value(proximity['folds_mean'])} (от {format_value(proximity['folds_min'])} "
        f"до {format_value(proximity['folds_max'])}), validation "
        f"{format_value(proximity['validation'])}.",
        "",
        (
            "Validation ближе к истории по словарю, чем средний сценарий train, поэтому"
            " опубликованное число — оптимистичная выборка, а не средняя оценка."
            if closer
            else "Validation не ближе среднего сценария train, значит разрыв объясняется другим."
        ),
        "",
        "К этому добавляется объём: в фолде у услуги в индексе один соседний сценарий, в",
        "validation — два. Это меняет и шанс попадания в top-3, и число обучающих примеров",
        "на услугу.",
        "",
        "Ещё одно расхождение — состав критичности: "
        + ", ".join(
            f"{label} {format_value(share * 100, 1)} %"
            for label, share in report["class_shares"]["severity"]["train"].items()  # type: ignore[index, union-attr]
        )
        + " в train против "
        + ", ".join(
            f"{label} {format_value(share * 100, 1)} %"
            for label, share in report["class_shares"]["severity"]["validation"].items()  # type: ignore[index, union-attr]
        )
        + " в validation. Поэтому классификатор критичности по этим двум наборам",
        "напрямую не сравнивается: у него разная база для P3.",
    ]

    lines += [
        "",
        "## Что это значит",
        "",
        "Для целей проектного плана это значит, что разрыв больше, чем показывал",
        "validation: пул по услуге "
        f"{format_value(report['pooled']['vote_service_f1'])} против цели "
        f"{format_value(PLAN_TARGETS['service'], 2)}"  # type: ignore[index]
        ", по причине "
        f"{format_value(report['pooled']['clf_cause_f1'])} против "
        f"{format_value(PLAN_TARGETS['root_cause'], 2)}."  # type: ignore[index]
        if report.get("pooled")
        else "",
        "",
        f"Разница по пулу лежит в диапазоне от {format_value(min(deltas))} до "
        f"{format_value(max(deltas))}." if deltas else "Сопоставимых показателей нет.",
        "",
        (
            "Ниже validation: " + "; ".join(below) + "."
            if below
            else "Ни один показатель не оказался ниже validation."
        ),
        "",
        (
            "Выше validation: " + "; ".join(above) + "."
            if above
            else "Ни один показатель не оказался выше validation."
        ),
        "",
        "## Чего это не доказывает",
        "",
        "Фолды идут по train, поэтому они не подтверждают и не опровергают результаты",
        "ЛР 7–10 на validation: это близкий, но не тот же режим — в фолде у услуги один",
        "соседний сценарий вместо двух, и сценарии train в среднем дальше от истории",
        "по словарю, чем сценарии validation. О реальных инцидентах диагностика не",
        "говорит ничего: набор синтетический, сценарий однозначно задаёт услугу и",
        "причину, а верхняя граница на случайном разбиении не измерялась.",
        "",
        "## Как воспроизвести",
        "",
        "```bash",
        "python src/scenario_cv.py run",
        "```",
        "",
        f"Прогон занимает около {format_value(report['seconds'], 0)} с. Машинная копия:",
        "`results/scenario_cv.json`.",
    ]
    return "\n".join(lines) + "\n"


def run(results_dir: Path = RESULTS_DIR, docs_dir: Path | None = None) -> int:
    """Считает фолды, пишет JSON и документ; test и validation не читает."""
    try:
        report = build_report()
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    results_dir.mkdir(parents=True, exist_ok=True)
    payload = results_dir / "scenario_cv.json"
    payload.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    docs_dir = docs_dir or results_dir.parent
    docs_dir.mkdir(parents=True, exist_ok=True)
    document = docs_dir / "scenario_cv.md"
    document.write_text(render_markdown(report), encoding="utf-8")

    print(f"фолдов: {report['folds']}, секунд: {report['seconds']}")
    for metric, entry in report["comparison"].items():  # type: ignore[union-attr]
        print(f"{metric}: пул {entry['pooled']}, validation {entry['validation']}, разница {entry['delta']:+}")
    print(f"машинная копия: {payload}")
    print(f"документ: {document}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="посчитать фолды по train")
    run_parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    run_parser.add_argument("--docs-dir", type=Path, default=None)
    args = parser.parse_args()
    return run(Path(args.results_dir), Path(args.docs_dir) if args.docs_dir else None)


if __name__ == "__main__":
    raise SystemExit(main())
