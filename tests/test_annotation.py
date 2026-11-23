"""Слепые листы пилотной разметки: выборка, лист без меток и ключ.

Лист показывает разметчику только номер карточки и описание, а ключ с
`incident_id` и сценарием лежит отдельно. CLI пишет три файла и не затирает
заполненные листы без --force.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import annotation_sheet as sheets

LABEL_COLUMNS = ("severity", "service", "root_cause")


@pytest.fixture(scope="module")
def validation(default_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Кусок validation с гарантированным покрытием всех семи сценариев."""
    return default_splits["validation"].groupby("scenario_id", sort=True).head(17).reset_index(
        drop=True
    )


@pytest.fixture
def source(validation: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "validation.csv"
    validation.to_csv(path, index=False, encoding="utf-8")
    return path


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["annotation_sheet.py", *args])
    return sheets.main()


def _cli_args(source: Path, outdir: Path, count: int, repeat: int) -> list[str]:
    return [
        "--sources",
        str(source),
        "--outdir",
        str(outdir),
        "--count",
        str(count),
        "--repeat",
        str(repeat),
        "--seed",
        "5",
    ]


def test_sample_has_requested_size(validation: pd.DataFrame) -> None:
    assert len(sheets.sample_cards(validation, 30, seed=7)) == 30


def test_sample_covers_every_scenario(validation: pd.DataFrame) -> None:
    """Стратификация: при выборке не меньше числа сценариев представлены все."""
    sample = sheets.sample_cards(validation, 30, seed=7)

    assert set(sample["scenario_id"]) == set(validation["scenario_id"])


def test_sample_is_deterministic_for_one_seed(validation: pd.DataFrame) -> None:
    first = sheets.sample_cards(validation, 30, seed=7)
    second = sheets.sample_cards(validation, 30, seed=7)

    assert first["incident_id"].tolist() == second["incident_id"].tolist()


def test_sample_takes_cards_only_from_the_given_frame(validation: pd.DataFrame) -> None:
    sample = sheets.sample_cards(validation, 30, seed=7)

    assert set(sample["incident_id"]) <= set(validation["incident_id"])
    assert sample["incident_id"].is_unique


def test_sample_rejects_count_above_the_frame(validation: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=str(len(validation))):
        sheets.sample_cards(validation, len(validation) + 1, seed=7)


def test_sheet_does_not_leak_dataset_labels(validation: pd.DataFrame) -> None:
    """В листе нет ни меток набора, ни служебных колонок с ответом."""
    sample = sheets.sample_cards(validation, 30, seed=7)
    sheet, _ = sheets.build_sheet(sample, "A", seed=7)

    for column in LABEL_COLUMNS:
        assert column in sheet.columns
        assert sheet[column].eq("").all(), column
    for column in ("scenario_id", "event_id", "incident_id"):
        assert column not in sheet.columns, column


def test_sheet_cards_are_unique_and_prefixed(validation: pd.DataFrame) -> None:
    sample = sheets.sample_cards(validation, 30, seed=7)
    sheet, key = sheets.build_sheet(sample, "A", seed=7)

    assert sheet["card"].tolist() == [f"A{number:03d}" for number in range(1, 31)]
    assert sheet["card"].is_unique
    assert key["card"].is_unique


def test_key_maps_cards_to_selected_incidents(validation: pd.DataFrame) -> None:
    sample = sheets.sample_cards(validation, 30, seed=7)
    sheet, key = sheets.build_sheet(sample, "A", seed=7)

    assert set(key["card"]) == set(sheet["card"])
    assert set(key["incident_id"]) == set(sample["incident_id"])
    assert key["incident_id"].is_unique
    known = sample.set_index("incident_id")["scenario_id"]
    assert key.set_index("incident_id")["scenario_id"].to_dict() == known.to_dict()


def test_sheet_keeps_descriptions_verbatim(validation: pd.DataFrame) -> None:
    sample = sheets.sample_cards(validation, 30, seed=7)
    sheet, key = sheets.build_sheet(sample, "A", seed=7)
    merged = sheet.merge(key, on="card", validate="one_to_one").set_index("incident_id")

    expected = sample.set_index("incident_id")["description"]

    assert merged["description"].to_dict() == expected.loc[merged.index].to_dict()


def test_cli_writes_three_files(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "annotation"

    code = _run(monkeypatch, *_cli_args(source, outdir, count=30, repeat=10))

    assert code == 0
    assert sorted(path.name for path in outdir.iterdir()) == [
        "form_a.html",
        "form_b.html",
        "key.csv",
        "pass_a.csv",
        "pass_b.csv",
    ]
    first = pd.read_csv(outdir / "pass_a.csv", dtype=str, keep_default_na=False)
    second = pd.read_csv(outdir / "pass_b.csv", dtype=str, keep_default_na=False)
    key = pd.read_csv(outdir / "key.csv", dtype=str, keep_default_na=False)
    assert len(first) == 30
    assert len(second) == 10
    assert len(key) == 40
    assert set(key["pass_name"]) == {"a", "b"}


def test_cli_second_pass_is_a_subset_of_the_first(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Повторный проход берёт карточки первого, но нумерует их заново."""
    outdir = tmp_path / "annotation"

    assert _run(monkeypatch, *_cli_args(source, outdir, count=30, repeat=10)) == 0

    key = pd.read_csv(outdir / "key.csv", dtype=str, keep_default_na=False)
    first_key = key[key["pass_name"] == "a"]
    second_key = key[key["pass_name"] == "b"]
    first = pd.read_csv(outdir / "pass_a.csv", dtype=str, keep_default_na=False)
    second = pd.read_csv(outdir / "pass_b.csv", dtype=str, keep_default_na=False)

    assert set(second_key["incident_id"]) <= set(first_key["incident_id"])
    assert second_key["incident_id"].is_unique
    assert set(second["card"]).isdisjoint(set(first["card"]))


def test_cli_does_not_overwrite_without_force(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "annotation"
    args = _cli_args(source, outdir, count=30, repeat=10)
    assert _run(monkeypatch, *args) == 0
    filled = outdir / "pass_a.csv"
    filled.write_text("заполненный лист", encoding="utf-8")

    assert _run(monkeypatch, *args) == 1
    assert filled.read_text(encoding="utf-8") == "заполненный лист"


def test_cli_overwrites_with_force(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "annotation"
    args = _cli_args(source, outdir, count=30, repeat=10)
    assert _run(monkeypatch, *args) == 0
    (outdir / "pass_a.csv").write_text("заполненный лист", encoding="utf-8")

    assert _run(monkeypatch, *args, "--force") == 0

    rebuilt = pd.read_csv(outdir / "pass_a.csv", dtype=str, keep_default_na=False)
    assert len(rebuilt) == 30
    assert rebuilt.columns[:2].tolist() == ["card", "description"]


def test_cli_fails_when_count_exceeds_the_frame(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "annotation"

    code = _run(monkeypatch, *_cli_args(source, outdir, count=10_000, repeat=10))

    assert code == 1
    assert not (outdir / "pass_a.csv").exists()


def test_cli_fails_when_validation_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "annotation"
    args = _cli_args(tmp_path / "нет_такого_файла.csv", outdir, count=30, repeat=10)

    assert _run(monkeypatch, *args) == 1
    assert not outdir.exists()

def test_form_embeds_cards_without_labels(source: Path, tmp_path: Path) -> None:
    """HTML-форма содержит все карточки, но ни одной метки набора."""
    outdir = tmp_path / "annotation"
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    sample = sheets.sample_cards(frame, 12, 7)
    markup = sheets.build_form(sheets.build_sheet(sample, "A", 7)[0], "Проход A", "pass_a.csv", "key")

    payload = markup.split("const CARDS = ", 1)[1].split(";\n", 1)[0]
    cards = json.loads(payload)

    assert len(cards) == 12
    assert all(set(card) == {"card", "description"} for card in cards)
    assert 'link.download = "pass_a.csv"' in markup
    assert "STORAGE_KEY" in markup
    assert "$" not in markup


def test_cli_writes_forms_next_to_the_sheets(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "annotation"

    code = _run(monkeypatch, *_cli_args(source, outdir, count=30, repeat=10))

    assert code == 0
    for name in ("form_a.html", "form_b.html"):
        text = (outdir / name).read_text(encoding="utf-8")
        assert "const CARDS = " in text
        assert "Скачать CSV" in text


def test_forms_only_keeps_filled_sheets(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--forms-only пересобирает формы из готовых листов и не трогает CSV."""
    outdir = tmp_path / "annotation"
    assert _run(monkeypatch, *_cli_args(source, outdir, count=12, repeat=4)) == 0
    first = pd.read_csv(outdir / "pass_a.csv", dtype=str, keep_default_na=False)
    first.loc[0, "severity"] = "P1"
    first.to_csv(outdir / "pass_a.csv", index=False, encoding="utf-8-sig")

    code = _run(monkeypatch, "--forms-only", "--outdir", str(outdir))

    assert code == 0
    kept = pd.read_csv(outdir / "pass_a.csv", dtype=str, keep_default_na=False)
    assert kept.loc[0, "severity"] == "P1"
    assert "Таблица решений" in (outdir / "form_a.html").read_text(encoding="utf-8")
