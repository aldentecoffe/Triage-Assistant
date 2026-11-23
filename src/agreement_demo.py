"""Демонстрация расчёта согласия на 120 карточках: метки синтетические.

Это не пилот и не разметка. Метки первого прохода взяты из набора, второй проход
повторяет их с детерминированными изменениями: часть критичности сдвинута на
уровень, часть сервисов заменена на соседний, часть причин сведена к `change`.
Так проверяется, что `agreement.py` считает долю совпадений, каппу, сверку
с набором и журнал расхождений на объёме, близком к пилотному.

Метки синтетические: их нельзя выдавать за ответы разметчика. Результат
воспроизводится командой `python src/agreement_demo.py`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from agreement import ALLOWED, combine, collect, discrepancies, report

SOURCE = "data/processed/incidents_validation_v1.1.csv"
COUNT = 120

# Детерминированные изменения второго прохода, чтобы демонстрация была
# воспроизводимой без ГПСЧ и не зависела от порядка запуска.
SEVERITY_STEP = 9
SERVICE_STEP = 15
CAUSE_STEP = 21
SEVERITY_DOWN = {"P1": "P2", "P2": "P3", "P3": "P4", "P4": "P3"}
SERVICE_SWAP = {
    "billing": "bss",
    "bss": "billing",
    "platform": "database",
    "database": "platform",
    "transport": "transport",
    "ims": "ims",
    "external": "external",
}


def build_sheets(source: str, count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает два прохода, ключ и метки набора на выбранных карточках."""
    frame = pd.read_csv(source, dtype=str, keep_default_na=False).head(count)
    sheet = pd.DataFrame(
        {
            "card": [f"D{number:03d}" for number in range(1, len(frame) + 1)],
            "description": frame["description"].tolist(),
            "severity": frame["severity"].tolist(),
            "service": frame["service"].tolist(),
            "root_cause": frame["root_cause"].tolist(),
            "confidence": ["3"] * len(frame),
            "comment": [""] * len(frame),
        }
    )

    changed = sheet.copy()
    for index in range(len(changed)):
        if index % SEVERITY_STEP == 0:
            changed.loc[index, "severity"] = SEVERITY_DOWN[changed.loc[index, "severity"]]
        if index % SERVICE_STEP == 0:
            changed.loc[index, "service"] = SERVICE_SWAP[changed.loc[index, "service"]]
        if index % CAUSE_STEP == 0:
            changed.loc[index, "root_cause"] = "change"

    incident_ids = [f"I{number:03d}" for number in range(1, len(frame) + 1)]
    key = pd.concat(
        [
            pd.DataFrame({"card": sheet["card"], "incident_id": incident_ids, "pass_name": "a"}),
            pd.DataFrame({"card": changed["card"], "incident_id": incident_ids, "pass_name": "b"}),
        ],
        ignore_index=True,
    )
    gold = pd.DataFrame(
        {
            "incident_id": incident_ids,
            **{column: sheet[column].tolist() for column in ALLOWED},
        }
    )
    return collect(sheet), collect(changed), key, gold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--count", type=int, default=COUNT)
    args = parser.parse_args()

    try:
        first, second, key, gold = build_sheets(args.source, args.count)
    except OSError as error:
        print(f"Ошибка чтения {args.source}: {error}")
        return 1

    combined = combine(first, second, key, gold)
    print("Демонстрация расчёта. Метки синтетические, это не пилот.")
    print()
    print(report(combined))
    print()
    log = discrepancies(combined)
    print(f"Расхождений в демонстрационном журнале: {len(log)}")
    print(log["label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
