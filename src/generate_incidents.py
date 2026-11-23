"""Генератор синтетических инцидентов версии 1.1.

Создаёт детерминированный набор по каталогу сценариев из incident_taxonomy,
добавляет контрольные дефекты и пишет сырой CSV. Критичность выводится
из формулировки последствия по таблице IMPACT_SEVERITY.

Запуск:
    python src/generate_incidents.py --count 6000
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from incident_taxonomy import (
    CASCADES,
    DATASET_VERSION,
    DEFECTS,
    GENERATOR_VERSION,
    IMPACT_SEVERITY,
    MONITORING_SOURCES,
    REGIONS,
    REQUIRED_COLUMNS,
    RU_SHARE,
    SCENARIOS,
    SPLIT_SHARES,
)
from text_utils import text_hash

SEED = 20261123
EPOCH = datetime(2026, 8, 1, 0, 0, 0)
TIME_SPAN_DAYS = 110

# Доля записей сценария, уходящая в каскадные события, и потолок таких событий на пару.
CASCADE_RECORD_SHARE = 4
CASCADE_EVENTS_PER_PAIR = 3

# Единицы, которые в тексте не могут быть дробными.
INTEGER_UNITS = {"шт", "шт/мин", "шт/с", "шт/час", "сессий", "хопов", "дней"}

UNIT_EN = {
    "шт": "",
    "шт/мин": "/min",
    "шт/с": "/s",
    "шт/час": "/h",
    "мс": " ms",
    "мин": " min",
    "с": " s",
    "хопов": " hops",
    "сессий": " sessions",
    "запросов/с": " req/s",
    "дней": " days",
    "%": " %",
}

EMPTY_DESCRIPTION_VARIANTS = ("", "   ", " \t")


def _fmt_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{int(round(value)):,}".replace(",", " ")
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}".replace(".", ",")


def draw_value(threshold: int, unit: str, direction: str, rng: random.Random) -> float:
    """Значение метрики по ту сторону порога и в физических границах."""
    if direction == "high":
        ceiling = 100.0 if unit == "%" and threshold <= 100 else None
        raw = threshold * rng.uniform(1.05, 1.4 if ceiling else 1.9)
        if ceiling is not None:
            raw = min(ceiling, raw)
        if unit in INTEGER_UNITS:
            return float(max(int(round(raw)), int(threshold) + 1))
        if abs(threshold) >= 1000:
            return float(round(raw, -1))
        if abs(threshold) >= 10:
            return float(round(raw))
        return round(raw, 1)

    if direction != "low":
        raise ValueError(f"неизвестное направление метрики: {direction}")
    if threshold <= 0:
        return float(threshold - rng.randint(1, 10))
    raw = threshold * rng.uniform(0.2, 0.85)
    if unit in INTEGER_UNITS:
        return float(max(0, min(int(round(raw)), int(threshold) - 1)))
    if abs(threshold) >= 10:
        return float(round(raw))
    return round(raw, 1)


def compose_text(
    scenario: dict, host: str, ci: str, rng: random.Random
) -> tuple[str, str, str, str]:
    """Собирает заголовок и описание; возвращает (title, description, language, severity)."""
    metric_name, threshold, unit, direction = rng.choice(scenario["metrics"])
    value = draw_value(threshold, unit, direction, rng)
    value_text = _fmt_number(value)
    threshold_text = _fmt_number(threshold)
    unit_text = f" {unit}" if unit else ""
    unit_en = UNIT_EN[unit]

    symptom_ru = rng.choice(scenario["symptoms_ru"])
    impact_index = rng.randrange(len(scenario["impacts_ru"]))
    impact_ru = scenario["impacts_ru"][impact_index]
    language = "ru" if rng.random() < RU_SHARE else "en"
    severity_by_impact = IMPACT_SEVERITY[scenario["id"]]
    severity = (
        severity_by_impact["ru"][impact_index] if language == "ru" else severity_by_impact["en"]
    )

    if language == "ru":
        symptom = symptom_ru[0].upper() + symptom_ru[1:]
        impact = impact_ru[0].upper() + impact_ru[1:]
        frames = (
            f"{symptom}. {metric_name} = {value_text}{unit_text}, "
            f"порог {threshold_text}{unit_text}. {impact}. Объект {ci}.",
            f"{symptom} на {host}. {metric_name} держится {value_text}{unit_text} "
            f"при пороге {threshold_text}{unit_text}. {impact}. Объект {ci}.",
            f"{symptom}; {metric_name} {value_text}{unit_text}. {impact}. "
            f"{ci}, узел {host}.",
        )
        description = rng.choice(frames)
        title = f"{symptom}: {metric_name}={value_text}{unit_text}"
    else:
        symptom_en = scenario["symptom_en"]
        impact_en = scenario["impact_en"]
        value_en = value_text.replace(",", ".")
        threshold_en = threshold_text.replace(",", ".")
        description = (
            f"{symptom_en[0].upper()}{symptom_en[1:]}. {metric_name} is "
            f"{value_en}{unit_en} against threshold {threshold_en}{unit_en}. "
            f"{impact_en[0].upper()}{impact_en[1:]}. Item {ci}."
        )
        title = f"{symptom_en[0].upper()}{symptom_en[1:]}: {metric_name}={value_en}{unit_en}"

    return title, description, language, severity


def allocate_counts(total: int) -> dict[str, int]:
    """Делит записи по split и сценариям так, чтобы сумма совпала с total."""
    raw = {split: total * share for split, share in SPLIT_SHARES.items()}
    per_split = {split: int(value) for split, value in raw.items()}
    remainder = total - sum(per_split.values())
    by_fraction = sorted(raw, key=lambda split: raw[split] - per_split[split], reverse=True)
    for split in by_fraction[:remainder]:
        per_split[split] += 1

    counts: dict[str, int] = {}
    for split, records in per_split.items():
        in_split = [s["id"] for s in SCENARIOS if s["split"] == split]
        base, rest = divmod(records, len(in_split))
        for index, scenario_id in enumerate(in_split):
            counts[scenario_id] = base + (1 if index < rest else 0)
    return counts


def minimum_total() -> int:
    """Наименьшее total, при котором каждому сценарию достаётся хотя бы запись."""
    for total in range(1, 1000):
        if min(allocate_counts(total).values()) >= 1:
            return total
    raise RuntimeError("не удалось подобрать минимальный размер набора")


def _new_event(index: list[int]) -> str:
    index[0] += 1
    return f"E{index[0]:06d}"


def _draw_time(rng: random.Random) -> datetime:
    return EPOCH + timedelta(days=rng.randint(0, TIME_SPAN_DAYS), minutes=rng.randint(0, 1439))


def build_rows(total: int = 6000, seed: int = SEED) -> list[dict]:
    """Возвращает сырые записи вместе с контрольными дефектами."""
    required = minimum_total()
    if total < required:
        raise ValueError(
            f"нужно минимум {required} записей: по одной на сценарий, запрошено {total}"
        )

    rng = random.Random(seed)
    scenarios = {s["id"]: s for s in SCENARIOS}
    counts = allocate_counts(total)
    event_index = [0]
    ci_counter = [100000 + rng.randrange(100000)]
    records: list[dict] = []

    def next_ci() -> str:
        """Уникальный CI: описания не совпадают случайно, дубли остаются только внесённые."""
        ci_counter[0] += rng.randint(7, 97)
        return f"CI-{ci_counter[0]:06d}"

    def make_record(scenario: dict, event_id: str, started: datetime) -> dict:
        region_ru, region_code = rng.choice(REGIONS)
        host = f"host-{region_code}{rng.randint(1, 200):03d}"
        ci = next_ci()
        title, description, language, severity = compose_text(scenario, host, ci, rng)
        created = started + timedelta(minutes=rng.randint(0, 30))
        return {
            "incident_id": "",
            "scenario_id": scenario["id"],
            "event_id": event_id,
            "created_at": created.strftime("%Y-%m-%dT%H:%M"),
            "source_system": rng.choice(MONITORING_SOURCES),
            "region": region_ru,
            "affected_ci": ci,
            "title": title,
            "description": description,
            "language": language,
            "severity": severity,
            "service": scenario["service"],
            "root_cause": scenario["root_cause"],
            "split": scenario["split"],
            "rights_status": "demo_allowed",
            "synthetic": "true",
            "source": f"generated:src/generate_incidents.py@{GENERATOR_VERSION}",
            "generator_version": GENERATOR_VERSION,
            "dataset_version": DATASET_VERSION,
            "text_hash": text_hash(description),
        }

    # Каскады: одно событие даёт записи в двух сервисах одного split.
    for left_id, right_id in CASCADES:
        available = min(
            counts[left_id] // CASCADE_RECORD_SHARE,
            counts[right_id] // CASCADE_RECORD_SHARE,
            CASCADE_EVENTS_PER_PAIR,
        )
        for _ in range(available):
            event_id = _new_event(event_index)
            started = _draw_time(rng)
            for _ in range(rng.randint(2, 3)):
                records.append(make_record(scenarios[left_id], event_id, started))
                counts[left_id] -= 1
            for _ in range(rng.randint(1, 2)):
                records.append(make_record(scenarios[right_id], event_id, started))
                counts[right_id] -= 1

    # Остальные записи: часть объединяется в события по 2-3 записи.
    for scenario_id, remaining in counts.items():
        scenario = scenarios[scenario_id]
        event_id = ""
        started = EPOCH
        size_left = 0
        while remaining > 0:
            if size_left == 0:
                event_id = _new_event(event_index)
                started = _draw_time(rng)
                size_left = rng.randint(2, 3) if rng.random() < 0.3 else 1
            records.append(make_record(scenario, event_id, started))
            size_left -= 1
            remaining -= 1

    records.extend(_inject_defects(records, rng))
    rng.shuffle(records)
    for number, record in enumerate(records, start=1):
        record["incident_id"] = f"I{number:06d}"

    _assert_no_natural_duplicates(records)
    for record in records:
        record.pop("_defect", None)
    return records


def _assert_no_natural_duplicates(records: list[dict]) -> None:
    """Страхует от регресса: валидные записи не должны совпадать по тексту."""
    natural = [r["text_hash"] for r in records if not r.get("_defect")]
    duplicates = len(natural) - len(set(natural))
    if duplicates:
        raise RuntimeError(f"генератор создал {duplicates} непредусмотренных дубликатов текста")


def _inject_defects(records: list[dict], rng: random.Random) -> list[dict]:
    """Добавляет дефектные записи внутрь уже пронумерованного набора."""
    needed = sum(DEFECTS.values())
    if len(records) < needed:
        raise ValueError(f"для контрольных дефектов нужно минимум {needed} записей")

    defects: list[dict] = []
    sources = rng.sample(records, needed)

    for source in sources[: DEFECTS["duplicate_text"]]:
        copy = dict(source)
        copy["incident_id"] = ""
        copy["_defect"] = True
        defects.append(copy)

    offset = DEFECTS["duplicate_text"]
    empty_count = DEFECTS["empty_description"]
    if empty_count > len(EMPTY_DESCRIPTION_VARIANTS):
        raise ValueError("не хватает вариантов пустого описания в EMPTY_DESCRIPTION_VARIANTS")
    for source, broken in zip(
        sources[offset:][:empty_count],
        EMPTY_DESCRIPTION_VARIANTS[:empty_count],
    ):
        copy = dict(source)
        copy["incident_id"] = ""
        copy["description"] = broken
        copy["title"] = ""
        copy["text_hash"] = text_hash(broken)
        copy["_defect"] = True
        defects.append(copy)

    offset += empty_count
    for source in sources[offset:][: DEFECTS["rights_unknown"]]:
        copy = dict(source)
        copy["incident_id"] = ""
        copy["rights_status"] = "unknown"
        copy["_defect"] = True
        defects.append(copy)

    return defects


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(REQUIRED_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=6000, help="целевое число валидных записей")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="data/raw/incidents_raw_v1.1.csv")
    args = parser.parse_args()

    try:
        rows = build_rows(total=args.count, seed=args.seed)
    except ValueError as error:
        print(f"Ошибка: {error}")
        return 1
    write_csv(rows, Path(args.out))
    defect_total = sum(DEFECTS.values())
    print(f"Записано {len(rows)} строк в {args.out}")
    print(f"  валидных записей: {len(rows) - defect_total}")
    print(f"  контрольных дефектов: {defect_total} {DEFECTS}")
    print(f"  версия набора: {DATASET_VERSION}, seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
