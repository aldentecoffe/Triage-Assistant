"""Краевые случаи пайплайна данных лабораторной 4.

Дополняют `tests/test_dataset_pipeline.py`: идемпотентность очистки, пустые
входы, граница порога длины описания, стабильность структуры по seed,
срабатывание проверок split и коды возврата CLI.

Наборы в тестах маленькие (140 записей): при таком объёме все семь сервисов
попадают в каждый split, а прогон занимает миллисекунды.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_incidents import build_rows, write_csv  # noqa: E402
from incident_taxonomy import (  # noqa: E402
    CASCADES,
    DEFECTS,
    REQUIRED_COLUMNS,
    SCENARIOS,
    SERVICES,
)
from prepare_dataset import MIN_TOKEN_COUNT, clean  # noqa: E402
from split_dataset import (  # noqa: E402
    SCENARIO_SPLIT,
    SPLIT_ORDER,
    assign,
    find_violations,
)
from text_utils import normalise, text_hash  # noqa: E402

# 140 записей: 84 в train и по 28 в validation и test, то есть по 6 и 4 записи
# на сценарий. Меньше минимума (minimum_total()) нельзя — часть сценариев
# останется без строк.
SMALL_TOTAL = 140
SEED = 20261123


@pytest.fixture(scope="module")
def small_raw() -> pd.DataFrame:
    return pd.DataFrame(build_rows(total=SMALL_TOTAL))


@pytest.fixture(scope="module")
def small_clean(small_raw: pd.DataFrame) -> pd.DataFrame:
    return clean(small_raw)[0]


@pytest.fixture(scope="module")
def small_assigned(small_clean: pd.DataFrame) -> pd.DataFrame:
    return assign(small_clean)


def _frame(*descriptions: str) -> pd.DataFrame:
    """Минимальный кадр для проверки очистки: только нужные колонки."""
    return pd.DataFrame(
        {
            "incident_id": [f"I{index:06d}" for index in range(1, len(descriptions) + 1)],
            "description": list(descriptions),
            "rights_status": ["demo_allowed"] * len(descriptions),
        }
    )


def _words(count: int) -> str:
    return " ".join(f"слово{index}" for index in range(count))


def _scenario_of(split: str) -> str:
    return next(scenario["id"] for scenario in SCENARIOS if scenario["split"] == split)


def _run_cli(module: str, *args: str) -> subprocess.CompletedProcess:
    """Запуск скрипта из src как отдельного процесса, как это делает пользователь."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / f"{module}.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )


# --- очистка: идемпотентность, пустые входы, границы -----------------------


def test_clean_is_idempotent(small_clean: pd.DataFrame) -> None:
    again, log = clean(small_clean)

    assert log.empty
    assert again.equals(small_clean)


def test_clean_partitions_input(small_raw: pd.DataFrame, small_clean: pd.DataFrame) -> None:
    _, log = clean(small_raw)

    kept_ids = set(small_clean["incident_id"])
    dropped_ids = set(log["incident_id"])
    assert len(small_clean) + len(log) == len(small_raw)
    assert kept_ids.isdisjoint(dropped_ids)
    assert kept_ids | dropped_ids == set(small_raw["incident_id"])


@pytest.mark.parametrize("dtype", ["object", "string", "str"])
def test_clean_accepts_empty_frame(dtype: str) -> None:
    empty = pd.DataFrame({column: pd.Series(dtype=dtype) for column in REQUIRED_COLUMNS})

    kept, log = clean(empty)

    assert kept.empty
    assert log.empty
    assert list(kept.columns) == list(REQUIRED_COLUMNS)
    assert list(log.columns) == [
        "incident_id",
        "stage",
        "reason",
        "description",
        "rights_status",
    ]


def test_clean_accepts_empty_frame_with_default_dtypes() -> None:
    empty = pd.DataFrame({"incident_id": [], "description": [], "rights_status": []})

    kept, log = clean(empty)

    assert kept.empty
    assert log.empty


def test_clean_keeps_exactly_at_token_threshold() -> None:
    frame = _frame(
        _words(MIN_TOKEN_COUNT - 1),
        _words(MIN_TOKEN_COUNT),
        f"  {_words(MIN_TOKEN_COUNT + 2)}\t",
    )

    kept, log = clean(frame)

    assert kept["incident_id"].tolist() == ["I000002", "I000003"]
    assert log["incident_id"].tolist() == ["I000001"]
    assert log["reason"].unique().tolist() == ["empty_or_short_description"]


def test_clean_drops_any_status_other_than_demo_allowed() -> None:
    frame = _frame(_words(7), _words(8), _words(9))
    frame.loc[1, "rights_status"] = "unknown"
    frame.loc[2, "rights_status"] = "restricted"

    kept, log = clean(frame)

    assert kept["incident_id"].tolist() == ["I000001"]
    assert log.set_index("incident_id")["stage"].to_dict() == {"I000002": 1, "I000003": 1}
    assert set(log["reason"]) == {"rights_unknown"}


def test_clean_preserves_input_order() -> None:
    shared = _words(7) + " из-за петли"
    frame = _frame(
        shared,  # 1 остаётся
        _words(6),  # 2 остаётся
        shared,  # 3 дубликат строки 1
        _words(8),  # 4 остаётся
        "   ",  # 5 пустое описание
        _words(5) + " BGP",  # 6 остаётся
    )

    kept, _ = clean(frame)

    assert kept["incident_id"].tolist() == ["I000001", "I000002", "I000004", "I000006"]
    assert kept.index.tolist() == [0, 1, 2, 3]


# --- генератор: независимость структуры от seed ----------------------------


@pytest.mark.parametrize("seed", [1, 7])
def test_generator_structure_does_not_depend_on_seed(
    seed: int,
    small_raw: pd.DataFrame,
    small_clean: pd.DataFrame,
    small_assigned: pd.DataFrame,
) -> None:
    frame = pd.DataFrame(build_rows(total=SMALL_TOTAL, seed=seed))

    kept, log = clean(frame)
    violations = find_violations(assign(kept))

    assert len(frame) == len(small_raw)
    assert len(kept) == len(small_clean)
    assert log["reason"].value_counts().to_dict() == {
        "duplicate_text": DEFECTS["duplicate_text"],
        "empty_or_short_description": DEFECTS["empty_description"],
        "rights_unknown": DEFECTS["rights_unknown"],
    }
    assert assign(kept)["split"].value_counts().to_dict() == (
        small_assigned["split"].value_counts().to_dict()
    )
    assert violations == []
    assert frame["description"].tolist() != small_raw["description"].tolist()


@pytest.mark.parametrize("total", [35, 1001])
def test_generator_returns_requested_number_of_valid_records(total: int) -> None:
    rows = build_rows(total=total)

    assert len(rows) - sum(DEFECTS.values()) == total


# --- split: запрет неизвестных сценариев и подмены -------------------------


def test_assign_rejects_unknown_scenario(small_clean: pd.DataFrame) -> None:
    frame = small_clean.copy()
    frame.loc[frame.index[0], "scenario_id"] = "SC-XXX-99"

    with pytest.raises(ValueError, match="SC-XXX-99"):
        assign(frame)


def test_assign_recomputes_split_from_catalogue(small_clean: pd.DataFrame) -> None:
    frame = small_clean.copy()
    scenario = _scenario_of("train")
    frame.loc[frame["scenario_id"].eq(scenario), "split"] = "test"

    assigned = assign(frame)

    assert assigned["split"].equals(assigned["scenario_id"].map(SCENARIO_SPLIT))
    assert find_violations(assigned) == []


def test_find_violations_detects_event_crossing(small_assigned: pd.DataFrame) -> None:
    frame = small_assigned.copy()
    frame.loc[frame["scenario_id"].eq(_scenario_of("train")), "event_id"] = "E-LEAK"
    frame.loc[frame["scenario_id"].eq(_scenario_of("test")), "event_id"] = "E-LEAK"

    violations = find_violations(frame)

    event_violations = [item for item in violations if item.startswith("event_id:")]
    assert len(event_violations) == 1
    assert "E-LEAK" in event_violations[0]


def test_find_violations_detects_text_hash_crossing(small_assigned: pd.DataFrame) -> None:
    frame = small_assigned.copy()
    frame.loc[frame["scenario_id"].eq(_scenario_of("train")), "text_hash"] = "f" * 64
    frame.loc[frame["scenario_id"].eq(_scenario_of("test")), "text_hash"] = "f" * 64

    violations = find_violations(frame)

    assert any(item.startswith("text_hash:") for item in violations)


def test_find_violations_detects_cascade_split_divergence(small_assigned: pd.DataFrame) -> None:
    left, right = CASCADES[0]
    frame = small_assigned.copy()
    moved_to = "validation" if SCENARIO_SPLIT[right] != "validation" else "test"
    frame.loc[frame["scenario_id"].eq(right), "split"] = moved_to

    violations = find_violations(frame)

    assert f"каскад {left} / {right} разъехался по split" in violations


def test_find_violations_reports_missing_services(small_assigned: pd.DataFrame) -> None:
    train_only = small_assigned[small_assigned["split"] == "train"].copy()

    violations = find_violations(train_only)

    assert len(violations) == 2
    assert all(item.startswith(("validation:", "test:")) for item in violations)
    assert all("нет сервисов" in item for item in violations)
    assert all(all(service in item for service in SERVICES) for item in violations)


# --- CLI: коды возврата и файлы -------------------------------------------


def test_generate_cli_writes_expected_rows(tmp_path: Path) -> None:
    output = tmp_path / "raw.csv"

    result = _run_cli("generate_incidents", "--count", str(SMALL_TOTAL), "--out", str(output))

    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert len(frame) == SMALL_TOTAL + sum(DEFECTS.values())
    assert list(frame.columns) == list(REQUIRED_COLUMNS)
    assert not frame["incident_id"].duplicated().any()


def test_prepare_cli_writes_clean_and_log(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    write_csv(build_rows(total=SMALL_TOTAL), raw_path)
    clean_path = tmp_path / "clean.csv"
    log_path = tmp_path / "log.csv"

    result = _run_cli(
        "prepare_dataset",
        "--raw",
        str(raw_path),
        "--out",
        str(clean_path),
        "--log",
        str(log_path),
    )

    assert result.returncode == 0, result.stderr
    cleaned = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
    log = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    assert len(cleaned) == SMALL_TOTAL
    assert log["reason"].value_counts().to_dict() == {
        "duplicate_text": DEFECTS["duplicate_text"],
        "empty_or_short_description": DEFECTS["empty_description"],
        "rights_unknown": DEFECTS["rights_unknown"],
    }
    assert sorted(log["stage"].unique().tolist()) == ["1", "2", "3"]


def test_split_cli_writes_three_files(tmp_path: Path) -> None:
    cleaned, _ = clean(pd.DataFrame(build_rows(total=SMALL_TOTAL)))
    clean_path = tmp_path / "clean.csv"
    cleaned.to_csv(clean_path, index=False, encoding="utf-8")
    outdir = tmp_path / "processed"

    result = _run_cli("split_dataset", "--clean", str(clean_path), "--outdir", str(outdir))

    assert result.returncode == 0, result.stderr
    parts = {
        split: pd.read_csv(
            outdir / f"incidents_{split}_v1.0.csv", dtype=str, keep_default_na=False
        )
        for split in SPLIT_ORDER
    }
    assert {split: len(part) for split, part in parts.items()} == {
        "train": 84,
        "validation": 28,
        "test": 28,
    }
    combined = pd.concat(parts.values(), ignore_index=True)
    assert set(combined["incident_id"]) == set(cleaned["incident_id"])
    assert (combined["split"] == combined["scenario_id"].map(SCENARIO_SPLIT)).all()


def test_split_cli_returns_one_on_event_crossing(tmp_path: Path) -> None:
    cleaned, _ = clean(pd.DataFrame(build_rows(total=SMALL_TOTAL)))
    frame = assign(cleaned)
    frame.loc[frame["scenario_id"].eq(_scenario_of("train")), "event_id"] = "E-LEAK"
    frame.loc[frame["scenario_id"].eq(_scenario_of("test")), "event_id"] = "E-LEAK"
    clean_path = tmp_path / "clean.csv"
    frame.to_csv(clean_path, index=False, encoding="utf-8")
    outdir = tmp_path / "processed"

    result = _run_cli("split_dataset", "--clean", str(clean_path), "--outdir", str(outdir))

    assert result.returncode == 1
    assert "Нарушения разделения" in result.stdout
    assert "E-LEAK" in result.stdout
    assert not outdir.exists()


def test_split_cli_ignores_tampered_split_column(tmp_path: Path) -> None:
    cleaned, _ = clean(pd.DataFrame(build_rows(total=SMALL_TOTAL)))
    tampered = cleaned.copy()
    tampered.loc[tampered["scenario_id"].eq(_scenario_of("train")), "split"] = "test"
    clean_path = tmp_path / "clean.csv"
    tampered.to_csv(clean_path, index=False, encoding="utf-8")
    outdir = tmp_path / "processed"

    result = _run_cli("split_dataset", "--clean", str(clean_path), "--outdir", str(outdir))

    # split в файле не источник истины: assign() пересчитывает его по каталогу,
    # поэтому подмена колонки не даёт ни нарушения, ни кода 1.
    assert result.returncode == 0, result.stderr
    combined = pd.concat(
        [
            pd.read_csv(
                outdir / f"incidents_{split}_v1.0.csv", dtype=str, keep_default_na=False
            )
            for split in SPLIT_ORDER
        ],
        ignore_index=True,
    )
    assert (combined["split"] == combined["scenario_id"].map(SCENARIO_SPLIT)).all()


# --- нормализация и хеш текста --------------------------------------------


def test_normalise_lowercases_and_collapses_whitespace() -> None:
    assert normalise("  Падает\tBGP\n сессия  ") == "падает bgp сессия"
    assert normalise("") == ""
    assert normalise("  \n\t ") == ""


def test_normalise_keeps_punctuation() -> None:
    # Пунктуация сохраняется намеренно: так дубликат отличается от похожей фразы.
    assert normalise("BGP, сессия; Idle.") == "bgp, сессия; idle."
    assert text_hash("a, b") != text_hash("a b")


def test_text_hash_ignores_case_and_whitespace() -> None:
    assert text_hash("BGP-сессия упала") == text_hash("bgp-сессия   упала\n")


def test_text_hash_of_blank_text_is_sha256_of_empty_string() -> None:
    empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    assert hashlib.sha256(b"").hexdigest() == empty_hash
    assert text_hash("") == empty_hash
    assert text_hash("   ") == empty_hash


def test_text_hash_is_lowercase_sha256_hex() -> None:
    value = text_hash("падает сессия")

    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")
    assert value == hashlib.sha256("падает сессия".encode("utf-8")).hexdigest()
