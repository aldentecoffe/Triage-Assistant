"""Согласие разметчиков: доля совпадений, каппа Коэна и журнал расхождений.

Две таблицы. Первая — первый проход против второго, только по карточкам, которые
размечались дважды: она отвечает, устойчива ли инструкция. Вторая — каждый проход
против меток набора: первый проход берётся целиком (120 карточек), второй — свои
30. Сверка с набором считается на всех размеченных карточках, иначе 90 из 120
выпадают из главного вывода. «Спорный» и пустые ячейки в каппу не входят,
но считаются отдельно.

Запуск из корня репозитория:
    python src/agreement.py
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


def read_key(path: str | Path) -> pd.DataFrame:
    """Читает ключ «номер карточки → incident_id» и проверяет его структуру."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"ключ {path} пуст")
    required = {"card", "incident_id", "pass_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"в ключе {path} нет колонок {sorted(missing)}")
    duplicates = frame.duplicated(subset=["pass_name", "card"])
    if duplicates.any():
        cards = frame.loc[duplicates, "card"].tolist()[:5]
        raise ValueError(f"в ключе повторяются номера карточек: {cards}")
    return frame


def prepare_side(sheet: pd.DataFrame, key: pd.DataFrame, side: str) -> pd.DataFrame:
    """Добавляет incident_id и переименовывает метки и состояния в *_a или *_b."""
    mapping = key[key["pass_name"] == side].set_index("card")["incident_id"]
    prepared = sheet.assign(incident_id=sheet["card"].map(mapping))
    missing = prepared.loc[prepared["incident_id"].isna(), "card"].tolist()
    if missing:
        prefix = "" if side == "a" else f"{side.upper()}:"
        raise ValueError(f"номеров карточек нет в ключе: {[prefix + card for card in missing]}")
    return prepared.rename(
        columns={
            **{f"{column}_label": f"{column}_label_{side}" for column in ALLOWED},
            **{f"{column}_state": f"{column}_state_{side}" for column in ALLOWED},
        }
    )


def merge_passes(first: pd.DataFrame, second: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Сопоставляет первый и второй проход по номеру карточки через ключ."""
    left = prepare_side(first, key, "a")
    right = prepare_side(second, key, "b")
    keep = ["incident_id", *[f"{column}_label_b" for column in ALLOWED], *[f"{column}_state_b" for column in ALLOWED]]
    merged = left.merge(right[keep], on="incident_id", how="inner", validate="one_to_one")
    if len(merged) != len(right):
        raise ValueError("во втором листе есть карточки, которых нет в первом")
    return merged


def attach_gold(pairs: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Добавляет метки набора по incident_id."""
    reference = gold[["incident_id", *ALLOWED]].rename(
        columns={column: f"{column}_label_gold" for column in ALLOWED}
    )
    return pairs.merge(reference, on="incident_id", how="left", validate="many_to_one")


def combine(first: pd.DataFrame, second: pd.DataFrame, key: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Собирает полный первый проход с метками второго прохода и набора.

    Сверка с набором должна идти по всем размеченным карточкам первого прохода,
    а не только по тем тридцати, что повторялись: иначе 90 из 120 выпадают из
    главного вывода. Метки второго прохода остаются пустыми там, где карточку
    повторно не размечали.
    """
    left = prepare_side(first, key, "a")
    right = prepare_side(second, key, "b")
    keep = ["incident_id", *[f"{column}_label_b" for column in ALLOWED], *[f"{column}_state_b" for column in ALLOWED]]
    combined = left.merge(right[keep], on="incident_id", how="left", validate="one_to_one")
    combined = attach_gold(combined, gold)
    for column in ALLOWED:
        combined[f"{column}_state_b"] = combined[f"{column}_state_b"].fillna("empty")
    return combined


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


def report(combined: pd.DataFrame) -> str:
    """Собирает markdown-отчёт: согласие проходов и сверка каждого прохода с набором."""
    lines = []
    lines.append("### Согласие первого и второго прохода")
    lines.append("")
    lines.append("| Метка | Сравнено | Совпало | Доля совпадений | Каппа Коэна | Квадратичная каппа |")
    lines.append("|---|---|---|---|---|---|")
    for column, name in LABEL_NAMES.items():
        stats = agreement(combined, column, "a", "b")
        matched = round(stats["agreement"] * stats["n"]) if stats["n"] else 0
        lines.append(
            f"| {name} | {stats['n']} | {matched} | {format_share(stats['agreement'])} | "
            f"{format_share(stats['kappa'])} | {format_share(stats['weighted'])} |"
        )

    lines.append("")
    lines.append("### Сверка с метками набора")
    lines.append("")
    lines.append("| Метка | Проход A: сравнено | Проход A: доля | Проход B: сравнено | Проход B: доля |")
    lines.append("|---|---|---|---|---|")
    for column, name in LABEL_NAMES.items():
        first = agreement(combined, column, "a", "gold")
        second = agreement(combined, column, "b", "gold")
        lines.append(
            f"| {name} | {first['n']} | {format_share(first['agreement'])} | "
            f"{second['n']} | {format_share(second['agreement'])} |"
        )

    lines.append("")
    lines.append(
        "Спорных решений в первом проходе: "
        + ", ".join(
            f"{name} — {int(combined[f'{column}_state_a'].eq('unsure').sum())}"
            for column, name in LABEL_NAMES.items()
        )
    )
    return "\n".join(lines)


def discrepancies(combined: pd.DataFrame) -> pd.DataFrame:
    """Журнал расхождений: проходы разошлись или ответ не совпал с набором."""
    rows = []
    for _, row in combined.iterrows():
        for column, name in LABEL_NAMES.items():
            first = row[f"{column}_label_a"]
            second = row[f"{column}_label_b"]
            reference = row[f"{column}_label_gold"]
            state_a = row[f"{column}_state_a"]
            state_b = row[f"{column}_state_b"]
            if state_a == "empty" and state_b == "empty":
                continue
            differ = bool(pd.notna(first) and pd.notna(second) and first != second)
            unsure = "unsure" in (state_a, state_b)
            against_gold = bool(
                pd.notna(reference)
                and any(pd.notna(label) and label != reference for label in (first, second))
            )
            if not (differ or unsure or against_gold):
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
        key = read_key(folder / "key.csv")
        answers = pd.read_csv(args.data, dtype=str, keep_default_na=False)
        missing = {"incident_id", *ALLOWED} - set(answers.columns)
        if missing:
            raise ValueError(f"в {args.data} нет колонок {sorted(missing)}")
        gold = answers[["incident_id", *ALLOWED]]
        combined = combine(first, second, key, gold)
    except (OSError, ValueError, KeyError) as error:
        print(f"Ошибка: {error}")
        return 1

    print(report(combined))
    log = discrepancies(combined)
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
