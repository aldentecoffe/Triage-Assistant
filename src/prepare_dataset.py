"""Очистка сырого набора инцидентов по фиксированному порядку проверок.

Порядок важен: сначала отсекаются записи без подтверждённого допуска,
затем записи без пригодного описания и только потом дубликаты по хешу текста.
Иначе пустые описания попали бы в отчёт как дубликаты друг друга.

Запуск:
    python src/prepare_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from incident_taxonomy import REQUIRED_COLUMNS
from text_utils import MIN_TOKEN_COUNT, text_hash

CHECKS = (
    (1, "rights_unknown"),
    (2, "empty_or_short_description"),
    (3, "duplicate_text"),
)


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает очищенный набор и журнал исключений."""
    frame = frame.copy()
    frame["description"] = frame["description"].astype("string").fillna("")
    frame["text_hash"] = frame["description"].map(text_hash)
    frame["_order"] = range(len(frame))
    excluded: list[pd.DataFrame] = []

    allowed = frame["rights_status"].eq("demo_allowed")
    excluded.append(frame.loc[~allowed].assign(stage=1, reason="rights_unknown"))
    frame = frame.loc[allowed]

    token_count = frame["description"].str.strip().str.split().str.len().fillna(0)
    informative = token_count.ge(MIN_TOKEN_COUNT)
    excluded.append(
        frame.loc[~informative].assign(stage=2, reason="empty_or_short_description")
    )
    frame = frame.loc[informative]

    duplicated = frame.duplicated(subset="text_hash", keep="first")
    excluded.append(frame.loc[duplicated].assign(stage=3, reason="duplicate_text"))
    frame = frame.loc[~duplicated]

    log = pd.concat(excluded, ignore_index=True)
    log = log[["incident_id", "stage", "reason", "description", "rights_status"]]
    # Запись, снятую за отсутствие допуска, нельзя хранить с полным текстом:
    # тот же скрипт пойдёт на реальную выгрузку.
    log.loc[log["stage"] == 1, "description"] = ""

    frame = frame.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return frame, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/incidents_raw_v1.1.csv")
    parser.add_argument("--out", default="data/interim/incidents_clean_v1.1.csv")
    parser.add_argument("--log", default="data/interim/cleaning_log_v1.1.csv")
    args = parser.parse_args()

    try:
        raw = pd.read_csv(args.raw, dtype=str, keep_default_na=False)
    except OSError as error:
        print(f"Ошибка чтения {args.raw}: {error}")
        return 1

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        print(f"Ошибка: в файле нет обязательных колонок {missing}")
        return 1

    cleaned, log = clean(raw)

    for path in (args.out, args.log):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False, encoding="utf-8")
    log.to_csv(args.log, index=False, encoding="utf-8")

    print(f"Сырых записей: {len(raw)}")
    print(f"Исключено: {len(log)}")
    for stage, reason in CHECKS:
        count = int((log["stage"] == stage).sum())
        print(f"  {stage}. {reason}: {count}")
    print(f"Очищенных записей: {len(cleaned)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
