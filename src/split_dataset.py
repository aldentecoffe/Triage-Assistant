"""Назначение split по группам и проверка пересечений.

Группа — сценарий отказа. Сценарий целиком уходит в один split, поэтому
текстовые серии одного сценария и события каскадов не могут попасть сразу
в train и test.

Запуск:
    python src/split_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from incident_taxonomy import CASCADES, SCENARIOS, SERVICES, SPLIT_SHARES
from text_utils import MIN_TOKEN_COUNT

SPLIT_ORDER = ("train", "validation", "test")
GROUP_KEYS = ("scenario_id", "event_id", "text_hash")
SCENARIO_SPLIT = {scenario["id"]: scenario["split"] for scenario in SCENARIOS}


def assign(frame: pd.DataFrame) -> pd.DataFrame:
    """Назначает split по каталогу сценариев.

    Колонка split во входном файле не читается: решение принимает каталог,
    иначе устаревший файл принёс бы разделение, расходящееся с таксономией.
    """
    frame = frame.copy()
    mapped = frame["scenario_id"].map(SCENARIO_SPLIT)
    missing = frame.loc[mapped.isna(), "scenario_id"].unique()
    if len(missing):
        raise ValueError(f"сценарии без split в каталоге: {sorted(missing)}")
    frame = frame.drop(columns=["split"], errors="ignore")
    frame["split"] = mapped
    return frame


def find_violations(frame: pd.DataFrame) -> list[str]:
    """Ищет утечки, следы неочищенных данных и сценарии без split."""
    violations: list[str] = []

    if (frame["rights_status"] != "demo_allowed").any():
        count = int((frame["rights_status"] != "demo_allowed").sum())
        violations.append(f"записей без подтверждённого допуска: {count}")

    token_count = frame["description"].astype("string").str.strip().str.split().str.len()
    too_short = token_count.fillna(0).lt(MIN_TOKEN_COUNT)
    if too_short.any():
        violations.append(f"описаний короче {MIN_TOKEN_COUNT} токенов: {int(too_short.sum())}")

    duplicates = frame["text_hash"].duplicated()
    if duplicates.any():
        violations.append(f"дубликатов текста: {int(duplicates.sum())}")

    for key in GROUP_KEYS:
        grouped = frame.assign(_group_key=frame[key].fillna("<нет>"))
        spread = grouped.groupby("_group_key")["split"].nunique()
        crossing = spread[spread > 1]
        if len(crossing):
            examples = ", ".join(str(value) for value in crossing.index[:5])
            violations.append(f"{key}: {len(crossing)} групп в разных split ({examples})")

    split_by_scenario = frame.groupby("scenario_id")["split"].first().to_dict()
    for left, right in CASCADES:
        left_split = split_by_scenario.get(left)
        right_split = split_by_scenario.get(right)
        if left_split is None or right_split is None:
            continue
        if left_split != right_split:
            violations.append(f"каскад {left} / {right} разъехался по split")

    for split in SPLIT_ORDER:
        part = frame[frame["split"] == split]
        missing_services = set(SERVICES) - set(part["service"])
        if missing_services:
            violations.append(f"{split}: нет сервисов {sorted(missing_services)}")
    return violations


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        part = frame[frame["split"] == split]
        rows.append(
            {
                "split": split,
                "records": len(part),
                "scenarios": part["scenario_id"].nunique(),
                "events": part["event_id"].nunique(),
                "services": part["service"].nunique(),
                "root_causes": part["root_cause"].nunique(),
            }
        )
    rows.append(
        {
            "split": "итого",
            "records": len(frame),
            "scenarios": frame["scenario_id"].nunique(),
            "events": frame["event_id"].nunique(),
            "services": frame["service"].nunique(),
            "root_causes": frame["root_cause"].nunique(),
        }
    )
    return pd.DataFrame(rows)


def print_markdown(frame: pd.DataFrame) -> None:
    summary = summarise(frame)
    print("| Выборка | Записей | Сценариев | Событий | Сервисов | Категорий причин |")
    print("|---|---|---|---|---|---|")
    for _, row in summary.iterrows():
        print(
            f"| {row['split']} | {row['records']} | {row['scenarios']} | "
            f"{row['events']} | {row['services']} | {row['root_causes']} |"
        )

    print()
    print("| Выборка | P1 | P2 | P3 | P4 |")
    print("|---|---|---|---|---|")
    for split in SPLIT_ORDER:
        part = frame[frame["split"] == split]
        shares = (part["severity"].value_counts(normalize=True) * 100).round(1)
        cells = " | ".join(f"{shares.get(label, 0.0):.1f}%" for label in ("P1", "P2", "P3", "P4"))
        print(f"| {split} | {cells} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default="data/interim/incidents_clean_v1.0.csv")
    parser.add_argument("--outdir", default="data/processed")
    args = parser.parse_args()

    try:
        clean = pd.read_csv(args.clean, dtype=str, keep_default_na=False)
    except OSError as error:
        print(f"Ошибка чтения {args.clean}: {error}")
        return 1

    try:
        frame = assign(clean)
    except ValueError as error:
        print(f"Нарушение: {error}")
        return 1

    violations = find_violations(frame)
    if violations:
        print("Нарушения разделения:")
        for item in violations:
            print(f"  - {item}")
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for split, share in SPLIT_SHARES.items():
        part = frame[frame["split"] == split]
        part.to_csv(outdir / f"incidents_{split}_v1.0.csv", index=False, encoding="utf-8")
        print(f"{split}: {len(part)} записей (плановая доля {share:.0%})")

    print()
    print_markdown(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
