"""Разведочный анализ и проверки качества набора инцидентов версии 1.1.

Скрипт отвечает на вопросы лабораторной 5: объём и пропуски, поддержка
классов, дубликаты и почти-дубликаты, пересечение выборок, перенос словаря
и служебные поля, которые могут подсказывать метку. Результат - таблица
проверок и восемь графиков; каждый график отвечает на вопрос, от которого
зависит решение по данным или по модели.

Запуск из корня репозитория:
    python src/eda_incidents.py

Скрипт не пересобирает данные: сначала пройдите пайплайн
    python src/generate_incidents.py --count 6000
    python src/prepare_dataset.py
    python src/split_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from incident_taxonomy import (  # noqa: E402
    CASCADES,
    DEFECTS,
    IMPACT_SEVERITY,
    REQUIRED_COLUMNS,
    ROOT_CAUSES,
    SCENARIOS,
    SEVERITIES,
    SERVICES,
)
from text_utils import MIN_TOKEN_COUNT, text_hash  # noqa: E402

SPLIT_ORDER = ("train", "validation", "test")
LABEL_COLUMNS = ("severity", "service", "root_cause")
GROUP_KEYS = ("scenario_id", "event_id", "text_hash")
AUDIT_FIELDS = ("source_system", "region", "language", "host_region", "ci_decile", "scenario_id")
EVENT_TIME_LIMIT = 30  # минут между карточками одного события
MIN_SPLIT_CLASS_SUPPORT = 30  # меньше записей на класс в выборке - оценка по классу шумная
PROXY_WARN_LEVEL = 0.20  # Cramer's V, выше которого поле годится как подсказка метки

EXPECTED_CLEAN = 6000  # объём версии 1.1, закреплённый в Data Card и тестах

DATA_PATHS = {
    "raw": "data/raw/incidents_raw_v1.1.csv",
    "clean": "data/interim/incidents_clean_v1.1.csv",
    "log": "data/interim/cleaning_log_v1.1.csv",
    "train": "data/processed/incidents_train_v1.1.csv",
    "validation": "data/processed/incidents_validation_v1.1.csv",
    "test": "data/processed/incidents_test_v1.1.csv",
}

TOKEN_RE = re.compile(r"[a-zа-яё]+")
CI_RE = re.compile(r"ci-\d+")
HOST_RE = re.compile(r"host-([a-z]+)\d+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


# --- загрузка ---------------------------------------------------------------


def load_frames(root: Path) -> dict[str, pd.DataFrame]:
    """Читает сырой набор, очищенный, журнал исключений и три выборки."""
    paths = {name: root / path for name, path in DATA_PATHS.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "нет файлов данных: "
            + ", ".join(missing)
            + ". Пройдите пайплайн: generate_incidents.py -> prepare_dataset.py -> split_dataset.py"
        )
    frames = {
        name: pd.read_csv(path, dtype=str, keep_default_na=False)
        for name, path in paths.items()
    }
    frames["log"]["stage"] = frames["log"]["stage"].astype(int)
    return frames


def combined(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Очищенный набор, собранный из трёх выборок."""
    return pd.concat([frames[split] for split in SPLIT_ORDER], ignore_index=True)


# --- вспомогательные вычисления --------------------------------------------


def tokenize(text: object) -> list[str]:
    """Слова из текста в нижнем регистре: латиница и кириллица."""
    return TOKEN_RE.findall(str(text).lower())


def token_count(texts: pd.Series) -> pd.Series:
    """Длина в токенах по пробелам - та же мера, что в очистке и схеме.

    Регулярка выше режет подчёркивания и числа, поэтому для проверки длины
    она не подходит: порог MIN_TOKEN_COUNT в пайплайне считается по пробелам.
    """
    return texts.map(lambda value: len(str(value).split()))


def template_key(text: object) -> str:
    """Слепок шаблона: числа, номера CI и имена узлов заменены заглушками.

    Позволяет отличить точную копию записи от другой карточки, собранной
    по той же формулировке с другим значением метрики.
    """
    value = str(text).lower()
    value = CI_RE.sub("<ci>", value)
    value = HOST_RE.sub("host-<r>", value)
    value = NUMBER_RE.sub("<n>", value)
    value = re.sub(r"[^a-zа-яё<>]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def host_region(description: object) -> str:
    """Код региона из имени узла; пустая строка, если имени нет."""
    match = HOST_RE.search(str(description))
    return match.group(1) if match else ""


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Сила связи двух категориальных полей: 0 - независимы, 1 - функция."""
    table = pd.crosstab(left, right).to_numpy(dtype=float)
    total = table.sum()
    if total == 0 or min(table.shape) < 2:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    chi2 = float(((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())
    return float(math.sqrt(chi2 / (total * (min(table.shape) - 1))))


def vocabulary(texts: pd.Series) -> set[str]:
    return {token for text in texts for token in tokenize(text)}


def token_coverage(reference: set[str], texts: pd.Series) -> tuple[float, float]:
    """Доля известных токенов и доля карточек, где известны не меньше 80 %."""
    known = 0
    total = 0
    covered_docs = 0
    for text in texts:
        tokens = tokenize(text)
        total += len(tokens)
        matches = sum(token in reference for token in tokens)
        known += matches
        if tokens and matches / len(tokens) >= 0.8:
            covered_docs += 1
    share = known / total if total else 0.0
    doc_share = covered_docs / len(texts) if len(texts) else 0.0
    return share, doc_share


def lexical_overlap(train: pd.DataFrame, other: pd.DataFrame) -> tuple[float, float]:
    return token_coverage(vocabulary(train["description"]), other["description"])


# --- проверки ---------------------------------------------------------------


def check(name: str, value: str, decision: str, status: str = "ok") -> dict[str, str]:
    return {"check": name, "value": value, "decision": decision, "status": status}


def ru(value: float, digits: int = 1) -> str:
    """Число с десятичной запятой: в документах проекта так принято."""
    return f"{value:.{digits}f}".replace(".", ",")


def _empty(name: str) -> list[dict[str, str]]:
    """Заглушка для пустого входа: молчаливое «ок» здесь опаснее отказа."""
    return [check(name, "набор пуст", "проверка невозможна на пустом наборе", "fail")]


def checks_volume(frames: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    raw, clean, log = frames["raw"], frames["clean"], frames["log"]
    if raw.empty or clean.empty:
        return _empty("Объём raw и clean")
    expected = len(raw) - sum(DEFECTS.values())
    defects = sum(DEFECTS.values())
    status = "ok" if len(clean) == expected == EXPECTED_CLEAN and len(log) == defects else "fail"
    if status == "ok":
        decision = f"ожидание {expected}; набор версии 1.1 воспроизводится"
    else:
        decision = (
            f"ожидание {expected}, в очищенном наборе {len(clean)}, в журнале {len(log)}: "
            "набор не совпадает с версией 1.1, нужна пересборка или новая версия Data Card"
        )
    return [
        check(
            "Объём raw и clean",
            f"{len(raw)} и {len(clean)} записей, исключено {len(log)}",
            decision,
            status,
        ),
    ]


def checks_schema(raw: pd.DataFrame, clean: pd.DataFrame) -> list[dict[str, str]]:
    if raw.empty or clean.empty:
        return _empty("Схема")
    expected = set(REQUIRED_COLUMNS)
    problems = []
    for name, frame in (("raw", raw), ("clean", clean)):
        if set(frame.columns) != expected:
            missing = sorted(expected - set(frame.columns))
            extra = sorted(set(frame.columns) - expected)
            problems.append(f"{name}: нет {missing}, лишние {extra}")
    return [
        check(
            "Схема",
            f"{len(clean.columns)} столбцов, расхождений с REQUIRED_COLUMNS: {len(problems)}",
            "схема v1.1 без изменений" if not problems else "; ".join(problems),
            "ok" if not problems else "fail",
        ),
    ]


def checks_required(clean: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Пустые обязательные поля")
    required = (
        "incident_id",
        "scenario_id",
        "event_id",
        "created_at",
        "title",
        "description",
        "language",
        "severity",
        "service",
        "root_cause",
        "split",
        "rights_status",
    )
    empty = {
        column: int(clean[column].astype("string").fillna("").str.strip().eq("").sum())
        for column in required
    }
    empty = {column: count for column, count in empty.items() if count}
    return [
        check(
            "Пустые обязательные поля",
            f"{sum(empty.values())} пустых значений в очищенном наборе",
            "заполнено полностью" if not empty else f"проблемы: {empty}",
            "ok" if not empty else "fail",
        ),
    ]


def checks_keys(clean: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Уникальность ключей и хешей")
    id_duplicates = int(clean["incident_id"].duplicated().sum())
    ci_duplicates = int(clean["affected_ci"].duplicated().sum())
    ci_empty = int(clean["affected_ci"].astype("string").fillna("").str.strip().eq("").sum())
    hash_mismatch = int((clean["description"].map(text_hash) != clean["text_hash"]).sum())
    problems = id_duplicates or ci_duplicates or ci_empty or hash_mismatch
    return [
        check(
            "Уникальность ключей и хешей",
            f"повторов incident_id: {id_duplicates}, affected_ci: {ci_duplicates}, "
            f"пустых affected_ci: {ci_empty}, расхождений text_hash: {hash_mismatch}",
            "ключи и хеши пригодны для группировки",
            "ok" if not problems else "fail",
        ),
    ]


def checks_duplicates(clean: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Точные дубли текста")
    duplicates = int(clean["text_hash"].duplicated().sum())
    return [
        check(
            "Точные дубли текста",
            f"{duplicates} повторов text_hash в очищенном наборе",
            "дубликаты снимаются очисткой",
            "ok" if duplicates == 0 else "fail",
        ),
    ]


def checks_allowed_values(clean: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Допустимые значения меток")
    allowed = {
        "severity": set(SEVERITIES),
        "service": set(SERVICES),
        "root_cause": set(ROOT_CAUSES),
        "split": set(SPLIT_ORDER),
        "language": {"ru", "en"},
        "rights_status": {"demo_allowed"},
        "synthetic": {"true"},
    }
    problems = []
    for column, values in allowed.items():
        extra = sorted(set(clean[column]) - values)
        if extra:
            problems.append(f"{column}: {extra}")
    return [
        check(
            "Допустимые значения меток",
            f"нарушений: {len(problems)}",
            "словари API повторяют значения схемы" if not problems else "; ".join(problems),
            "ok" if not problems else "fail",
        ),
    ]


def checks_cleaning_log(clean: pd.DataFrame, log: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Журнал исключений")
    expected = {
        1: DEFECTS["rights_unknown"],
        2: DEFECTS["empty_description"],
        3: DEFECTS["duplicate_text"],
    }
    stages = {stage: int((log["stage"] == stage).sum()) for stage in expected}
    reasons = log.drop_duplicates("stage").set_index("stage")["reason"].to_dict()
    leftover = int(clean["incident_id"].isin(set(log["incident_id"])).sum())
    unredacted = int(log.loc[log["stage"] == 1, "description"].str.strip().ne("").sum())
    problems = []
    if len(log) != sum(DEFECTS.values()):
        problems.append(f"в журнале {len(log)} строк вместо {sum(DEFECTS.values())}")
    if stages != expected:
        problems.append(f"этапы {stages} вместо {expected}")
    if leftover:
        problems.append(f"{leftover} исключённых записей остались в clean")
    if unredacted:
        problems.append(f"{unredacted} записей без прав сохранены с текстом")
    order = "; ".join(f"{stage}. {reasons.get(stage, '?')}: {stages[stage]}" for stage in sorted(stages))
    return [
        check(
            "Журнал исключений",
            order,
            "каждая запись получает одну причину по порядку проверок"
            if not problems
            else "; ".join(problems),
            "ok" if not problems else "fail",
        ),
    ]


def checks_split_crossings(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return _empty("Пересечение групп между выборками")
    problems = []
    for key in GROUP_KEYS:
        spread = frame.groupby(key, dropna=False)["split"].nunique()
        crossing = int((spread > 1).sum())
        if crossing:
            problems.append(f"{key}: {crossing} групп в разных выборках")
    return [
        check(
            "Пересечение групп между выборками",
            "0 нарушений" if not problems else "; ".join(problems),
            "групповой split подтверждён: сценарий, событие и текст не пересекаются"
            if not problems
            else "split нужно пересобрать, прежние метрики аннулируются",
            "ok" if not problems else "fail",
        ),
    ]


def checks_cascades(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return _empty("Каскады в одной выборке")
    split_by_scenario = frame.groupby("scenario_id")["split"].first().to_dict()
    broken = [
        f"{left} / {right}"
        for left, right in CASCADES
        if split_by_scenario.get(left) != split_by_scenario.get(right)
    ]
    services_per_event = frame.groupby("event_id")["service"].nunique()
    events = int((services_per_event > 1).sum())
    value = f"пар: {len(CASCADES)}, событий: {events}, разъехавшихся пар: {len(broken)}"
    if broken:
        value += f" ({'; '.join(broken)})"
    return [
        check(
            "Каскады в одной выборке",
            value,
            "каскад нельзя разделять: карточки описывают одно происшествие",
            "ok" if not broken else "fail",
        ),
    ]


def checks_coverage(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return _empty("Покрытие меток по выборкам")
    rows = []
    missing_services = []
    missing_causes = []
    for split in SPLIT_ORDER:
        part = frame[frame["split"] == split]
        absent_services = sorted(set(SERVICES) - set(part["service"]))
        absent_causes = sorted(set(ROOT_CAUSES) - set(part["root_cause"]))
        if absent_services:
            missing_services.append(f"{split}: {absent_services}")
        if absent_causes:
            missing_causes.append(f"{split}: {absent_causes}")
        rows.append(
            f"{split} - сервисов {part['service'].nunique()}, причин {part['root_cause'].nunique()}"
        )
    status = "fail" if missing_services else ("warn" if missing_causes else "ok")
    if missing_services:
        decision = (
            f"в выборке нет сервисов ({'; '.join(missing_services)}): проверьте split, "
            "метрики по этим сервисам не считаются"
        )
    elif missing_causes:
        decision = (
            f"по категории, которой нет в выборке ({'; '.join(missing_causes)}), метрика не считается: "
            "вывод по ней делается на кросс-валидации train"
        )
    else:
        decision = "покрытие полное"
    details = []
    if missing_services:
        details.append(f"нет сервисов ({'; '.join(missing_services)})")
    if missing_causes:
        details.append(f"нет категорий ({'; '.join(missing_causes)})")
    return [
        check(
            "Покрытие меток по выборкам",
            "; ".join(rows) + ("; " + "; ".join(details) if details else ""),
            decision,
            status,
        ),
    ]


def impact_severity_map() -> dict[str, list[tuple[str, str]]]:
    """Последствие сценария и его класс: по этому правилу выставляется severity."""
    mapping: dict[str, list[tuple[str, str]]] = {}
    for scenario in SCENARIOS:
        marks = IMPACT_SEVERITY[scenario["id"]]
        rows = list(zip(scenario["impacts_ru"], marks["ru"]))
        rows.append((scenario["impact_en"], marks["en"]))
        mapping[scenario["id"]] = rows
    return mapping


def severity_mismatches(frame: pd.DataFrame) -> pd.Series:
    """Карточки, где severity не совпадает с классом последствия из текста."""
    mapping = impact_severity_map()
    text = frame["description"].astype("string").str.lower().fillna("")

    def expected(row) -> str:
        for phrase, label in mapping.get(row["scenario_id"], []):
            if phrase.lower() in row["_text"]:
                return label
        return "<не найдено>"

    work = frame.assign(_text=text)
    found = work.apply(expected, axis=1)
    return found.ne(work["severity"])


def checks_events(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return _empty("Согласованность события")
    work = frame.assign(
        created_at=pd.to_datetime(frame["created_at"], format="%Y-%m-%dT%H:%M", errors="coerce")
    )
    # Сверка опирается на текст последствия, поэтому на урезанных рамках
    # из тестов, где описания нет, она пропускается.
    if {"description", "scenario_id"}.issubset(frame.columns):
        mismatches = int(severity_mismatches(frame).sum())
    else:
        mismatches = 0
    mixed_events = int((work.groupby("event_id")["severity"].nunique() > 1).sum())
    multi_service = int((work.groupby("event_id")["service"].nunique() > 2).sum())
    spans = work.groupby("event_id")["created_at"].max() - work.groupby("event_id")["created_at"].min()
    long_events = int((spans > pd.Timedelta(minutes=EVENT_TIME_LIMIT)).sum())
    unparsed = int(work["created_at"].isna().sum())
    sizes = work.groupby("event_id").size()
    problems = mismatches + long_events + unparsed + multi_service
    return [
        check(
            "Согласованность события",
            f"{work['event_id'].nunique()} событий, карточек в событии {int(sizes.min())}-{int(sizes.max())}, "
            f"расхождений severity с последствием: {mismatches}, событий с разным масштабом влияния: "
            f"{mixed_events}, сервисов больше двух: {multi_service}, "
            f"разброс времени больше {EVENT_TIME_LIMIT} мин: {long_events}, "
            f"нечитаемых дат: {unparsed}",
            "критичность карточки выводится из её последствия; карточки события не независимы, "
            "оценка и бутстрэп считаются по событиям и сценариям",
            "ok" if problems == 0 else "fail",
        ),
    ]


def checks_class_support(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return _empty("Поддержка классов")
    expected = {
        "severity": set(SEVERITIES),
        "service": set(SERVICES),
        "root_cause": set(ROOT_CAUSES),
    }
    support = {}
    for column in LABEL_COLUMNS:
        counts = frame[column].value_counts()
        support[column] = (int(counts.min()), float(counts.min() / len(frame)), str(counts.idxmin()))
    thin = []
    for column in LABEL_COLUMNS:
        for split in SPLIT_ORDER:
            part = frame.loc[frame["split"] == split, column]
            absent = sorted(expected[column] - set(part))
            counts = part.value_counts()
            small = sorted(counts[counts < MIN_SPLIT_CLASS_SUPPORT].index)
            if absent:
                thin.append(f"{split}/{column}: нет {absent}")
            if small:
                thin.append(f"{split}/{column}: мало {small}")
    text = "; ".join(
        f"{column}: минимум {count} ({ru(share * 100)} %, {value})"
        for column, (count, share, value) in support.items()
    )
    return [
        check(
            "Поддержка классов",
            text + ("; " + "; ".join(thin) if thin else ""),
            "F1-macro по всем классам, для P1 отдельно recall; отсутствующие в выборке классы "
            f"и классы меньше {MIN_SPLIT_CLASS_SUPPORT} записей оцениваются по кросс-валидации train",
            "ok" if not thin else "warn",
        ),
    ]


def template_groups(clean: pd.DataFrame) -> pd.DataFrame:
    """Группы записей с одинаковым шаблоном текста."""
    work = clean.assign(template=clean["description"].map(template_key))
    return work.groupby("template").agg(
        records=("incident_id", "size"),
        scenarios=("scenario_id", "nunique"),
        splits=("split", "nunique"),
    )


def checks_templates(clean: pd.DataFrame, groups: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Повтор шаблонов текста")
    series = int((groups["records"] > 1).sum())
    covered = int(groups.loc[groups["records"] > 1, "records"].sum())
    crossing = int((groups["splits"] > 1).sum())
    multi_scenario = int((groups["scenarios"] > 1).sum())
    status = "fail" if crossing else ("warn" if series else "ok")
    return [
        check(
            "Повтор шаблонов текста",
            f"{len(groups)} шаблонов на {len(clean)} записей, серий: {series}, "
            f"они покрывают {covered} записей ({ru(covered / len(clean) * 100)} %), "
            f"в разных выборках: {crossing}, на несколько сценариев: {multi_scenario}",
            "text_hash ловит только точные копии, а серии закрывает группировка по сценарию: "
            "при случайном разделении почти у каждой карточки был бы почти-двойник в train",
            status,
        ),
    ]


def checks_lexical_transfer(frames: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    train = frames["train"]
    if train.empty or frames["validation"].empty or frames["test"].empty:
        return _empty("Перенос словаря train на другие выборки")
    rows = []
    share_test = 0.0
    for split in ("validation", "test"):
        share, docs = lexical_overlap(train, frames[split])
        rows.append(
            f"{split}: {ru(share * 100)} % токенов, {ru(docs * 100)} % карточек с покрытием 80 %+"
        )
        if split == "test":
            share_test = share
    return [
        check(
            "Перенос словаря train на другие выборки",
            "; ".join(rows),
            "общий словарь есть, формулировки новые: перенос ожидается, но не гарантирован; "
            "проверяется моделью в ЛР 6",
            "ok" if share_test >= 0.5 else "warn",
        ),
    ]


def proxy_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Cramer's V между служебными полями и тремя метками."""
    numbers = pd.to_numeric(
        clean["affected_ci"].str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    if numbers.notna().all() and numbers.nunique() >= 10:
        decile = pd.qcut(numbers, 10, labels=False, duplicates="drop").astype("string")
    else:
        decile = pd.Series("нет данных", index=clean.index, dtype="string")
    work = clean.assign(host_region=clean["description"].map(host_region), ci_decile=decile)
    rows = []
    for field in AUDIT_FIELDS:
        row = {"field": field}
        for label in LABEL_COLUMNS:
            row[label] = round(cramers_v(work[field], work[label]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def checks_proxy(proxies: pd.DataFrame) -> list[dict[str, str]]:
    if proxies.empty:
        return _empty("Proxy-признаки в служебных полях")
    plain = proxies.loc[
        ~proxies["field"].isin(["scenario_id", "ci_decile"]), list(LABEL_COLUMNS)
    ]
    plain_max = float(plain.to_numpy().max())
    ci_row = proxies.loc[proxies["field"] == "ci_decile", list(LABEL_COLUMNS)].iloc[0]
    scenario_max = float(
        proxies.loc[proxies["field"] == "scenario_id", list(LABEL_COLUMNS)].to_numpy().max()
    )
    ci_max = float(ci_row.max())
    value = (
        f"source_system, region, language, host_region: не выше {ru(plain_max, 2)}; "
        f"дециль affected_ci: {ru(ci_row['service'], 2)} (service), "
        f"{ru(ci_row['root_cause'], 2)} (root_cause); scenario_id: {ru(scenario_max, 2)}"
    )
    decision = (
        "в модель идёт только описание: обычные служебные поля сигнала не несут, "
        "а упорядоченный номер CI и scenario_id подсказывают сценарий и в признаки не попадают"
    )
    status = "ok" if max(plain_max, ci_max) < PROXY_WARN_LEVEL else "warn"
    return [check("Proxy-признаки в служебных полях", value, decision, status)]


def checks_time(clean: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Период и покрытие по времени")
    stamps = pd.to_datetime(clean["created_at"], format="%Y-%m-%dT%H:%M", errors="coerce")
    span = int((stamps.max() - stamps.min()).days)
    week_index = stamps.dt.to_period("W")
    all_weeks = pd.period_range(week_index.min(), week_index.max(), freq="W")
    filled = int(week_index.nunique())
    empty = len(all_weeks) - filled
    return [
        check(
            "Период и покрытие по времени",
            f"{stamps.min():%Y-%m-%d} — {stamps.max():%Y-%m-%d}, {span} дней, "
            f"недель с данными: {filled}, без данных: {empty}",
            "недели покрыты без пропусков, разделение по времени выигрыша не даёт; "
            "равномерность по неделям видна на графике, время в признаки не входит",
            "ok" if empty == 0 else "warn",
        ),
    ]


def checks_text_length(clean: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Длина описания")
    lengths = token_count(clean["description"])
    short = int((lengths < MIN_TOKEN_COUNT).sum())
    return [
        check(
            "Длина описания",
            f"{int(lengths.min())}-{int(lengths.max())} токенов, медиана {lengths.median():.0f}, "
            f"короче {MIN_TOKEN_COUNT}: {short}",
            "усечение не нужно, словарь и n-граммы считаются по полному тексту",
            "ok" if short == 0 else "warn",
        ),
    ]


def checks_title(clean: pd.DataFrame) -> list[dict[str, str]]:
    if clean.empty:
        return _empty("Заголовок повторяет описание")
    contained = 0
    for title, description in zip(clean["title"], clean["description"]):
        title_tokens = set(tokenize(title))
        description_tokens = set(tokenize(description))
        if title_tokens and title_tokens <= description_tokens:
            contained += 1
    share = contained / len(clean)
    return [
        check(
            "Заголовок повторяет описание",
            f"{ru(share * 100)} % заголовков состоят только из слов описания",
            "title в признаки не входит: это сжатая копия того же сигнала, а не второй признак",
            "ok" if share > 0.9 else "warn",
        ),
    ]


def checks_split_files(frames: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    """Сверяет три файла выборок с очищенным набором.

    Без этой проверки устаревший или обрезанный файл выборки проходит молча:
    остальные проверки смотрят на наборы по отдельности.
    """
    clean = frames["clean"]
    parts = [frames[split] for split in SPLIT_ORDER]
    if clean.empty or any(part.empty for part in parts):
        return _empty("Состав выборок и clean")
    combined_frame = pd.concat(parts, ignore_index=True)
    clean_ids = set(clean["incident_id"])
    split_ids = set(combined_frame["incident_id"])
    missing = int((~clean["incident_id"].isin(split_ids)).sum())
    extra = int((~combined_frame["incident_id"].isin(clean_ids)).sum())
    repeats = int(combined_frame["incident_id"].duplicated().sum())
    sizes = {split: len(frames[split]) for split in SPLIT_ORDER}
    volume_ok = sum(sizes.values()) == len(clean)
    ok = missing == 0 and extra == 0 and repeats == 0 and volume_ok
    return [
        check(
            "Состав выборок и clean",
            f"train {sizes['train']}, validation {sizes['validation']}, test {sizes['test']}; "
            f"записей clean без выборки: {missing}, лишних в выборках: {extra}, "
            f"повторов между выборками: {repeats}",
            "выборки собраны из этого очищенного набора"
            if ok
            else "выборки не совпадают с clean: пересобрать split_dataset.py",
            "ok" if ok else "fail",
        ),
    ]


def run_checks(frames: dict[str, pd.DataFrame]) -> tuple[list[dict[str, str]], pd.DataFrame]:
    clean = frames["clean"]
    frame = combined(frames)
    groups = template_groups(clean)
    proxies = proxy_table(clean)
    results: list[dict[str, str]] = []
    results += checks_volume(frames)
    results += checks_split_files(frames)
    results += checks_schema(frames["raw"], clean)
    results += checks_required(clean)
    results += checks_keys(clean)
    results += checks_duplicates(clean)
    results += checks_allowed_values(clean)
    results += checks_cleaning_log(clean, frames["log"])
    results += checks_split_crossings(frame)
    results += checks_cascades(frame)
    results += checks_coverage(frame)
    results += checks_events(frame)
    results += checks_class_support(frame)
    results += checks_templates(clean, groups)
    results += checks_lexical_transfer(frames)
    results += checks_proxy(proxies)
    results += checks_time(clean)
    results += checks_text_length(clean)
    results += checks_title(clean)
    return results, groups


# --- графики ----------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cleaning_funnel(frames: dict[str, pd.DataFrame], out: Path) -> Path:
    raw, log = len(frames["raw"]), frames["log"]
    stages = [
        ("сырой набор", raw),
        ("права подтверждены", raw - int((log["stage"] == 1).sum())),
        ("описание пригодно", raw - int(log["stage"].isin([1, 2]).sum())),
        ("без дублей (clean)", len(frames["clean"])),
    ]
    labels = [label for label, _ in stages]
    values = [value for _, value in stages]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh(labels[::-1], values[::-1], color=("#b7c9d6", "#86a8bd", "#5e8aa6", "#3f6f8f"))
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 40, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("карточек инцидентов")
    ax.set_title(f"Очистка: {raw} -> {len(frames['clean'])}, у каждой записи одна причина")
    return _save(fig, out / "01_cleaning_funnel.png")


def plot_split_composition(frames: dict[str, pd.DataFrame], out: Path) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8))
    positions = np.arange(len(SPLIT_ORDER))
    records = [len(frames[split]) for split in SPLIT_ORDER]
    bars = left.bar(positions, records, 0.55, color="#3f6f8f")
    for bar, value in zip(bars, records):
        left.text(bar.get_x() + bar.get_width() / 2, value + 40, str(value), ha="center", fontsize=9)
    left.set_xticks(positions, SPLIT_ORDER)
    left.set_ylabel("карточек")
    left.set_ylim(0, max(records) * 1.2)
    left.set_title("Записи")

    shares = {
        severity: [
            float(frames[split]["severity"].eq(severity).mean() * 100) for split in SPLIT_ORDER
        ]
        for severity in SEVERITIES
    }
    colors = ("#8c2f39", "#c97b4a", "#7d9a6b", "#4d6a8a")
    width = 0.2
    for index, severity in enumerate(SEVERITIES):
        values = shares[severity]
        right.bar(positions + (index - 1.5) * width, values, width, label=severity, color=colors[index])
        if severity == "P1":
            for position, value in zip(positions - 1.5 * width, values):
                right.text(position, value + 0.5, ru(value), ha="center", fontsize=7.5)
    p1 = shares["P1"]
    right.set_xticks(positions, SPLIT_ORDER)
    right.set_ylabel("доля записей, %")
    right.set_ylim(0, max(v for values in shares.values() for v in values) * 1.3)
    right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    right.set_title(f"Критичность: P1 от {ru(min(p1))} % до {ru(max(p1))} %")
    right.legend(title="severity", ncols=4, fontsize=8, loc="upper center")
    return _save(fig, out / "02_split_composition.png")


def plot_service_cause(clean: pd.DataFrame, out: Path) -> Path:
    table = pd.crosstab(clean["service"], clean["root_cause"]).reindex(
        index=list(SERVICES), columns=list(ROOT_CAUSES), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    image = ax.imshow(table.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    top = int(table.to_numpy().max())
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = int(table.iat[row, column])
            ax.text(column, row, value, ha="center", va="center", fontsize=8,
                    color="white" if value > top * 0.6 else "#22303c")
    fig.colorbar(image, ax=ax, label="карточек")
    ax.set_title(f"Сервис и корневая причина: заполнено {int((table > 0).to_numpy().sum())} из {table.size} ячеек")
    return _save(fig, out / "03_service_cause_matrix.png")


def plot_text_length(clean: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.arange(10, 42, 1)
    for language, color in (("ru", "#3f6f8f"), ("en", "#c97b4a")):
        values = token_count(clean.loc[clean["language"] == language, "description"])
        ax.hist(values, bins=bins, alpha=0.6, color=color,
                label=f"{language}: медиана {values.median():.0f}, n={len(values)}")
    top = ax.get_ylim()[1] * 1.15
    ax.set_ylim(0, top)
    ax.axvline(MIN_TOKEN_COUNT + 0.5, color="#8c2f39", linestyle="--", linewidth=1)
    ax.text(MIN_TOKEN_COUNT + 0.8, top * 0.92, f"порог {MIN_TOKEN_COUNT} токенов", color="#8c2f39", fontsize=8)
    ax.set_xlabel("токенов в описании")
    ax.set_ylabel("карточек")
    lengths = token_count(clean["description"])
    median_ru = token_count(clean.loc[clean["language"] == "ru", "description"]).median()
    median_en = token_count(clean.loc[clean["language"] == "en", "description"]).median()
    ax.set_title(
        f"Длина: {int(lengths.min())}-{int(lengths.max())} токенов; английские карточки "
        f"длиннее на {median_en - median_ru:.0f}"
    )
    ax.legend(fontsize=8)
    return _save(fig, out / "04_text_length.png")


def plot_lexical_overlap(frames: dict[str, pd.DataFrame], out: Path) -> Path:
    train = frames["train"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4))

    overall = {
        split: lexical_overlap(train, frames[split])[0] * 100
        for split in ("validation", "test")
    }
    labels = list(overall)
    bars = left.bar(labels, list(overall.values()), 0.5, color=("#7d9a6b", "#c97b4a"))
    for bar, value in zip(bars, overall.values()):
        left.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{ru(value)} %", ha="center", fontsize=9)
    left.set_ylim(0, 100)
    left.set_ylabel("токенов, известных в train, %")
    left.set_title("Перенос словаря целиком")

    per_service = [
        lexical_overlap(train, frames["test"][frames["test"]["service"] == service])[0] * 100
        for service in SERVICES
    ]
    right.barh(list(SERVICES)[::-1], per_service[::-1], color="#5e8aa6")
    for index, value in enumerate(per_service[::-1]):
        right.text(min(value + 1, 96), index, f"{ru(value, 0)} %", va="center", fontsize=8)
    right.set_xlim(0, 100)
    right.set_xlabel("токенов test, известных в train, %")
    right.set_title(f"По сервисам: от {ru(min(per_service), 0)} % до {ru(max(per_service), 0)} %")
    return _save(fig, out / "05_lexical_overlap.png")


def plot_templates(groups: pd.DataFrame, clean: pd.DataFrame, out: Path) -> Path:
    sizes = groups["records"]
    buckets = [
        ("1 карточка", int((sizes == 1).sum())),
        ("2-5", int(((sizes >= 2) & (sizes <= 5)).sum())),
        ("6-20", int(((sizes >= 6) & (sizes <= 20)).sum())),
        ("21-50", int(((sizes >= 21) & (sizes <= 50)).sum())),
        ("51+", int((sizes > 50).sum())),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar([label for label, _ in buckets], [value for _, value in buckets], color="#3f6f8f")
    for bar, (_, value) in zip(bars, buckets):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center", fontsize=9)
    series = int((sizes > 1).sum())
    covered = int(sizes[sizes > 1].sum())
    ax.set_ylabel("шаблонов")
    ax.set_xlabel("карточек в группе одного шаблона")
    ax.set_ylim(0, max(value for _, value in buckets) * 1.2)
    ax.set_title(
        f"{len(groups)} шаблонов на {len(clean)} карточек: {series} серий покрывают {covered}"
    )
    crossing = int((groups["splits"] > 1).sum())
    ax.text(0.99, 0.95, f"шаблонов, пересекающих выборки: {crossing}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#8c2f39")
    return _save(fig, out / "06_template_groups.png")


def plot_proxies(proxies: pd.DataFrame, out: Path) -> Path:
    table = proxies.set_index("field")[list(LABEL_COLUMNS)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    image = ax.imshow(table.to_numpy(), cmap="OrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = float(table.iat[row, column])
            ax.text(column, row, ru(value, 2), ha="center", va="center", fontsize=9,
                    color="white" if value > 0.55 else "#22303c")
    fig.colorbar(image, ax=ax, label="Cramer's V")
    ax.set_title("Служебные поля и метка: выше 0,2 — готовый proxy-признак")
    return _save(fig, out / "07_proxy_signals.png")


def plot_time(frame: pd.DataFrame, out: Path) -> Path:
    stamps = pd.to_datetime(frame["created_at"], format="%Y-%m-%dT%H:%M", errors="coerce")
    weekly = (
        frame.assign(week=stamps.dt.to_period("W").dt.start_time)
        .groupby(["week", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(SPLIT_ORDER), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bottom = np.zeros(len(weekly))
    colors = {"train": "#3f6f8f", "validation": "#7d9a6b", "test": "#c97b4a"}
    for split in SPLIT_ORDER:
        ax.bar(weekly.index, weekly[split], bottom=bottom, width=5.5, label=split, color=colors[split])
        bottom += weekly[split].to_numpy()
    ax.set_ylabel("карточек в неделю")
    ax.set_title(
        f"Покрытие по времени: {len(weekly)} недель, средний размер {ru(bottom.mean(), 0)} карточек"
    )
    ax.legend(fontsize=8, ncols=3)
    return _save(fig, out / "08_time_coverage.png")


def build_figures(frames: dict[str, pd.DataFrame], groups: pd.DataFrame, figures_dir: Path) -> list[Path]:
    frame = combined(frames)
    return [
        plot_cleaning_funnel(frames, figures_dir),
        plot_split_composition(frames, figures_dir),
        plot_service_cause(frames["clean"], figures_dir),
        plot_text_length(frames["clean"], figures_dir),
        plot_lexical_overlap(frames, figures_dir),
        plot_templates(groups, frames["clean"], figures_dir),
        plot_proxies(proxy_table(frames["clean"]), figures_dir),
        plot_time(frame, figures_dir),
    ]


# --- вывод ------------------------------------------------------------------


def to_markdown(results: list[dict[str, str]]) -> str:
    lines = ["| Проверка | Факт | Решение | Статус |", "|---|---|---|---|"]
    labels = {"ok": "ок", "warn": "внимание", "fail": "проблема"}
    for row in results:
        lines.append(
            f"| {row['check']} | {row['value']} | {row['decision']} | {labels[row['status']]} |"
        )
    return "\n".join(lines)


def display_path(path: Path, root: Path) -> str:
    """Путь относительно корня, если он внутри; иначе как есть."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="корень репозитория")
    parser.add_argument("--figures", default="docs/lab5/figures")
    parser.add_argument("--checks", default="docs/lab5/quality_checks.csv")
    args = parser.parse_args(argv)

    root = Path(args.root)
    try:
        frames = load_frames(root)
    except FileNotFoundError as error:
        print(f"Ошибка: {error}")
        return 1

    results, groups = run_checks(frames)
    figures = build_figures(frames, groups, root / args.figures)

    checks_path = root / args.checks
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(checks_path, index=False, encoding="utf-8")

    print(to_markdown(results))
    print()
    print(f"графиков: {len(figures)}")
    for path in figures:
        print(f"  {display_path(path, root)}")
    print(f"таблица проверок: {display_path(checks_path, root)}")
    failed = [row for row in results if row["status"] == "fail"]
    warned = [row for row in results if row["status"] == "warn"]
    print(f"итого: {len(results) - len(failed) - len(warned)} ок, {len(warned)} внимание, {len(failed)} проблема")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
