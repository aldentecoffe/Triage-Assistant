"""Слепые листы пилотной разметки для ЛР 6.

Собирает выборку карточек из validation, перемешивает и раскладывает в листы,
где разметчику видны только номер карточки и описание. Метки, сервис, причина,
`scenario_id` и `event_id` в лист не попадают: иначе проверялась бы не
инструкция, а память о сгенерированных метках.

Два листа: первый проход по `--count` карточкам, второй (отложенный) — по
`--repeat` карточкам из первого листа, в другом порядке и с другими номерами.
Ключ «номер карточки → incident_id» лежит отдельно и открывается только после
второго прохода.

Запуск из корня репозитория:
    python src/annotation_sheet.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALIDATION_PATH = "data/processed/incidents_validation_v1.1.csv"
SHEET_DIR = "data/annotation"
SEED = 20261123

FILL_COLUMNS = ("severity", "service", "root_cause", "confidence", "comment")


def sample_cards(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Стратифицированная выборка: карточки по кругу, равномерно по сценариям."""
    if count < 1:
        raise ValueError(f"нужно хотя бы одну карточку, запрошено {count}")
    if frame.empty:
        raise ValueError("не из чего выбирать: пустая выборка")
    if count > len(frame):
        raise ValueError(f"в выборке {len(frame)} карточек, запрошено {count}")

    groups = [
        part.sample(frac=1.0, random_state=seed)
        for _, part in frame.groupby("scenario_id", sort=True)
    ]
    ordered: list[pd.DataFrame] = []
    for position in range(max(len(part) for part in groups)):
        for part in groups:
            if position < len(part):
                ordered.append(part.iloc[[position]])

    sample = pd.concat(ordered[:count]).sample(frac=1.0, random_state=seed)
    return sample.reset_index(drop=True)


def build_sheet(sample: pd.DataFrame, prefix: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает лист без меток и ключ соответствия номеров."""
    shuffled = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    key = pd.DataFrame(
        {
            "card": [f"{prefix}{number:03d}" for number in range(1, len(shuffled) + 1)],
            "incident_id": shuffled["incident_id"].tolist(),
            "scenario_id": shuffled["scenario_id"].tolist(),
        }
    )
    sheet = pd.DataFrame(
        {
            "card": key["card"],
            "description": shuffled["description"].tolist(),
        }
    )
    for column in FILL_COLUMNS:
        sheet[column] = ""
    return sheet, key


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """UTF-8 с BOM: Excel открывает русский текст без танцев с кодировкой."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", default=VALIDATION_PATH)
    parser.add_argument("--outdir", default=SHEET_DIR)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="перезаписать заполненные листы")
    args = parser.parse_args()

    paths = {name: Path(args.outdir) / name for name in ("pass_a.csv", "pass_b.csv", "key.csv")}
    if args.count < 1 or args.repeat < 1:
        print("Ошибка: --count и --repeat должны быть больше нуля")
        return 1
    if not args.force and any(path.exists() for path in paths.values()):
        print(f"Листы уже есть в {args.outdir}: перезапись только с --force")
        return 1

    try:
        frame = pd.read_csv(args.validation, dtype=str, keep_default_na=False)
    except OSError as error:
        print(f"Ошибка чтения {args.validation}: {error}")
        return 1

    try:
        sample = sample_cards(frame, args.count, args.seed)
    except ValueError as error:
        print(f"Ошибка: {error}")
        return 1

    sheet_a, key_a = build_sheet(sample, "A", args.seed)
    repeat = sample.sample(n=min(args.repeat, len(sample)), random_state=args.seed + 1)
    sheet_b, key_b = build_sheet(repeat, "B", args.seed + 1)

    write_csv(sheet_a, paths["pass_a.csv"])
    write_csv(sheet_b, paths["pass_b.csv"])
    key = pd.concat(
        [key_a.assign(pass_name="a"), key_b.assign(pass_name="b")], ignore_index=True
    )
    write_csv(key, paths["key.csv"])

    print(f"первый проход: {len(sheet_a)} карточек, сценариев {sample['scenario_id'].nunique()}")
    print(f"повторный проход: {len(sheet_b)} карточек из первого листа")
    print(f"файлы: {paths['pass_a.csv']}, {paths['pass_b.csv']}, ключ {paths['key.csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
