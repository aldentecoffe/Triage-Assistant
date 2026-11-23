"""Согласие разметчиков: доля совпадений, каппа Коэна и журнал расхождений.

Сравниваются две пары: первый проход со вторым (устойчивость инструкции при
отложенном повторе) и каждый проход с метками набора (воспроизводит ли человек
то, что записано в данных). «Спорный» и пустые ячейки в расчёт не входят,
но считаются отдельно: если спорных много, инструкция не готова.

Запуск из корня репозитория:
    python src/agreement.py --labels data/annotation/pass_a.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from incident_taxonomy import ROOT_CAUSES, SEVERITIES, SERVICES

ALLOWED = {
    "severity": SEVERITIES,
    "service": tuple(SERVICES),
    "root_cause": tuple(ROOT_CAUSES),
}
UNSURE_VALUES = {"спорный", "спорно", "не знаю", "не уверен", "?"}
LABEL_NAMES = {
    "severity": "Критичность",
    "service": "Сервис",
    "root_cause": "Категория причины",
}


def parse_label(raw: object, column: str) -> tuple[str | None, str]:
    """Возвращает метку и её состояние: ok, unsure, empty, invalid."""
    value = "" if raw is None else str(raw).strip()
    if not value:
        return None, "empty"
    lowered = value.lower()
    if lowered in UNSURE_VALUES:
        return None, "unsure"
    for label in ALLOWED[column]:
        if lowered == label.lower():
            return label, "ok"
    return None, "invalid"


def read_sheet(path: str | Path) -> pd.DataFrame:
    """Читает лист разметки, сохраняя пустые ячейки пустыми строками."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [column for column in ("card", "description", *ALLOWED) if column not in frame]
    if missing:
        raise ValueError(f"в листе {path} нет колонок {missing}")
    return frame


def collect(frame: pd.DataFrame) -> pd.DataFrame:
    """Разбирает заполненные ячейки: метка, состояние, блокирующая ошибка."""
    work = frame.copy()
    errors: list[str] = []
    for column in ALLOWED:
        parsed = work[column].map(lambda value, name=column: parse_label(value, name))
        work[f"{column}_label"] = [item[0] for item in parsed]
        work[f"{column}_state"] = [item[1] for item in parsed]
        invalid = work.loc[work[f"{column}_state"] == "invalid", "card"].tolist()
        if invalid:
            errors.append(f"{column}: недопустимые значения у карточек {invalid}")
    if errors:
        raise ValueError("; ".join(errors))
    return work


def merge_passes(first: pd.DataFrame, second: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Сопоставляет первый и второй проход по номеру карточки через ключ."""
    pairs = {}
    for side in ("a", "b"):
        mapping = key[key["pass_name"] == side].set_index("card")["incident_id"]
        pairs[side] = mapping

    first = first.assign(incident_id=first["card"].map(pairs["a"]))
    second = second.assign(incident_id=second["card"].map(pairs["b"]))
    missing = first.loc[first["incident_id"].isna(), "card"].tolist()
    missing += [f"B:{card}" for card in second.loc[second["incident_id"].isna(), "card"]]
    if missing:
        raise ValueError(f"номеров карточек нет в ключе: {missing}")

    right = second[
        ["incident_id", *[f"{column}_label" for column in ALLOWED], *[f"{column}_state" for column in ALLOWED]]
    ].rename(
        columns={
            **{f"{column}_label": f"{column}_label_b" for column in ALLOWED},
            **{f"{column}_state": f"{column}_state_b" for column in ALLOWED},
        }
    )
    left = first.rename(
        columns={
            **{f"{column}_label": f"{column}_label_a" for column in ALLOWED},
            **{f"{column}_state": f"{column}_state_a" for column in ALLOWED},
        }
    )
    merged = left.merge(right, on="incident_id", how="inner", validate="one_to_one")
    if len(merged) != len(right):
        raise ValueError("во втором листе есть карточки, которых нет в первом")
    return merged


def attach_gold(pairs: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Добавляет метки набора к строкам, где совпали оба прохода."""
    reference = gold[["incident_id", *ALLOWED]].rename(
        columns={column: f"{column}_label_gold" for column in ALLOWED}
    )
    return pairs.merge(reference, on="incident_id", how="left", validate="many_to_one")


def label_values(pairs: pd.DataFrame, column: str, side: str) -> pd.Series:
    """Колонка метки нужной стороны: a, b или gold."""
    name = f"{column}_label_{side}"
    if name not in pairs:
        raise KeyError(f"в таблице нет колонки {name}")
    return pairs[name]


def agreement(pairs: pd.DataFrame, column: str, left: str = "a", right: str = "b") -> dict[str, object]:
    """Доля совпадений и каппа Коэна по одной метке между двумя сторонами."""
    first = label_values(pairs, column, left)
    second = label_values(pairs, column, right)
    mask = first.notna() & second.notna()
    compared = int(mask.sum())
    if compared == 0:
        return {"n": 0, "agreement": float("nan"), "kappa": float("nan"), "weighted": float("nan")}
    matches = int((first[mask] == second[mask]).sum())
    weighted = float("nan")
    values = pd.concat([first[mask], second[mask]])
    if values.nunique() < 2:
        # Каппа на одном классе не определена: делить на 1 − p_e нечего.
        kappa = float("nan")
    else:
        kappa = cohen_kappa_score(first[mask], second[mask])
        if column == "severity":
            weighted = cohen_kappa_score(first[mask], second[mask], weights="quadratic")
    return {
        "n": compared,
        "agreement": matches / compared,
        "kappa": float(kappa),
        "weighted": float(weighted),
    }


def format_value(value: object) -> str:
    """Прочерк вместо пустой метки: NaN в журнале читается как ошибка."""
    return "—" if pd.isna(value) else str(value)


def format_share(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.3f}".replace(".", ",")


def report(pairs: pd.DataFrame, gold: pd.DataFrame) -> str:
    """Собирает markdown-отчёт по обоим проходам."""
    pairs = attach_gold(pairs, gold)
    lines = []
    lines.append("### Согласие первого и второго прохода")
    lines.append("")
    lines.append("| Метка | Сравнено | Совпало | Доля совпадений | Каппа Коэна |")
    lines.append("|---|---|---|---|---|")
    for column, name in LABEL_NAMES.items():
        stats = agreement(pairs, column, "a", "b")
        lines.append(
            f"| {name} | {stats['n']} | {round(stats['agreement'] * stats['n']) if stats['n'] else 0} | "
            f"{format_share(stats['agreement'])} | {format_share(stats['kappa'])} |"
        )

    lines.append("")
    lines.append("### Сверка с метками набора")
    lines.append("")
    lines.append("| Метка | Совпало с набором, проход A | Совпало с набором, проход B |")
    lines.append("|---|---|---|")
    for column, name in LABEL_NAMES.items():
        first = agreement(pairs, column, "a", "gold")
        second = agreement(pairs, column, "b", "gold")
        lines.append(f"| {name} | {format_share(first['agreement'])} | {format_share(second['agreement'])} |")

    lines.append("")
    lines.append(
        "Спорных решений в первом проходе: "
        + ", ".join(
            f"{name} — {int(pairs[f'{column}_state_a'].eq('unsure').sum())}"
            for column, name in LABEL_NAMES.items()
        )
    )
    return "\n".join(lines)


def discrepancies(pairs: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Журнал расхождений: где проходы не совпали или разошлись с набором."""
    rows = []
    source = attach_gold(pairs, gold)
    for _, row in source.iterrows():
        for column, name in LABEL_NAMES.items():
            first = row[f"{column}_label_a"]
            second = row[f"{column}_label_b"]
            reference = row[f"{column}_label_gold"]
            if "empty" in (row[f"{column}_state_a"], row[f"{column}_state_b"]):
                continue
            if first == second and (pd.isna(reference) or first == reference):
                continue
            rows.append(
                {
                    "card": row["card"],
                    "incident_id": row["incident_id"],
                    "label": name,
                    "first": first,
                    "second": second,
                    "dataset": reference,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/annotation")
    parser.add_argument("--data", default="data/processed/incidents_validation_v1.1.csv")
    args = parser.parse_args()

    folder = Path(args.labels)
    try:
        first = collect(read_sheet(folder / "pass_a.csv"))
        second = collect(read_sheet(folder / "pass_b.csv"))
        key = pd.read_csv(folder / "key.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    try:
        pairs = merge_passes(first, second, key)
    except ValueError as error:
        print(f"Ошибка: {error}")
        return 1

    try:
        answers = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    except OSError as error:
        print(f"Ошибка чтения {args.data}: {error}")
        return 1
    gold = answers[["incident_id", *ALLOWED]]

    print(report(pairs, gold))
    log = discrepancies(pairs, gold)
    print("")
    print(f"Расхождений в журнале: {len(log)}")
    for _, row in log.head(20).iterrows():
        print(
            f"  {row['card']}: {row['label']} — A {format_value(row['first'])}, "
            f"B {format_value(row['second'])}, набор {format_value(row['dataset'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
