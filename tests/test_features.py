"""Признаки лабораторной 6: словарь и IDF считаются только по train.

Утечка из validation и test проверяется с двух сторон: слова, которых нет
в train, не попадают в словарь, а замена текстов validation и test не меняет
ни словарь, ни веса. Отдельно проверяется отказ от пересекающихся выборок
и запись манифеста через CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prepare_features as features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = PROJECT_ROOT / "src" / "prepare_features.py"
SPLIT_NAMES = ("train", "validation", "test")


def _slice(frame: pd.DataFrame, per_service: int) -> pd.DataFrame:
    """Кусок выборки с покрытием всех сервисов: иначе find_violations ругается."""
    return frame.groupby("service", sort=True).head(per_service).reset_index(drop=True)


@pytest.fixture(scope="module")
def frames(default_splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Небольшие train, validation и test из настоящего набора."""
    return {
        "train": _slice(default_splits["train"], 12),
        "validation": _slice(default_splits["validation"], 4),
        "test": _slice(default_splits["test"], 4),
    }


def _write_splits(frames: dict[str, pd.DataFrame], folder: Path) -> dict[str, str]:
    """Раскладывает выборки в отдельные csv и возвращает пути."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, frame in frames.items():
        path = folder / f"{split}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        paths[split] = str(path)
    return paths


def test_vocabulary_contains_only_train_ngrams(frames: dict[str, pd.DataFrame]) -> None:
    """Каждый признак словаря встречается в текстах train по тому же анализатору."""
    vectorizer = features.fit_vectorizer(frames["train"])
    analyzer = vectorizer.build_analyzer()
    train_terms = {
        term for text in frames["train"][features.TEXT_COLUMN] for term in analyzer(text)
    }

    assert vectorizer.vocabulary_
    assert set(vectorizer.get_feature_names_out()) <= train_terms


def test_word_seen_only_in_validation_does_not_enter_vocabulary(
    frames: dict[str, pd.DataFrame],
) -> None:
    """Слово, которого нет в train, отсутствует в словаре, даже если им полон validation."""
    sentinel = "словотолькоизвалидации"
    changed = dict(frames)
    changed["validation"] = frames["validation"].assign(
        description=f"{sentinel} проверка утечки словаря"
    )

    vectorizer = features.fit_vectorizer(changed["train"])

    assert sentinel not in vectorizer.vocabulary_
    assert features.transform(vectorizer, changed["validation"]).shape[1] == len(
        vectorizer.vocabulary_
    )


def test_validation_and_test_texts_do_not_change_idf(frames: dict[str, pd.DataFrame]) -> None:
    """Замена описаний в validation и test не меняет ни словарь, ни idf_."""
    base_vectorizer, base_manifest = features.build(frames)
    changed = dict(frames)
    changed["validation"] = frames["validation"].assign(
        description="совершенно новый текст про ёмкость узла доступа"
    )
    changed["test"] = frames["test"].assign(
        description="другой невиданный текст про баланс абонента"
    )
    other_vectorizer, other_manifest = features.build(changed)

    assert base_manifest["vocabulary_sha256"] == other_manifest["vocabulary_sha256"]
    assert list(base_vectorizer.get_feature_names_out()) == list(
        other_vectorizer.get_feature_names_out()
    )
    np.testing.assert_array_equal(base_vectorizer.idf_, other_vectorizer.idf_)


def test_fit_is_deterministic(frames: dict[str, pd.DataFrame]) -> None:
    """Два обучения на одном train дают одинаковый словарь и одинаковые веса."""
    first = features.fit_vectorizer(frames["train"])
    second = features.fit_vectorizer(frames["train"])

    assert list(first.get_feature_names_out()) == list(second.get_feature_names_out())
    np.testing.assert_array_equal(first.idf_, second.idf_)


def test_build_reports_rows_vocabulary_and_hash(frames: dict[str, pd.DataFrame]) -> None:
    """Манифест описывает словарь, его отпечаток и число строк каждой выборки."""
    vectorizer, manifest = features.build(frames)

    assert manifest["rows"] == {split: len(frame) for split, frame in frames.items()}
    assert manifest["vocabulary_size"] == len(vectorizer.vocabulary_)
    assert manifest["vocabulary_sha256"] == features.vocabulary_hash(vectorizer)
    assert manifest["features"] == len(vectorizer.get_feature_names_out())
    assert manifest["text_column"] == features.TEXT_COLUMN
    for split, frame in frames.items():
        assert features.transform(vectorizer, frame).shape[0] == len(frame)


def test_load_splits_reads_three_frames(
    tmp_path: Path, frames: dict[str, pd.DataFrame]
) -> None:
    """Три файла читаются как есть, порядок описаний сохраняется."""
    paths = _write_splits(frames, tmp_path)
    loaded = features.load_splits(paths)

    assert set(loaded) == set(SPLIT_NAMES)
    for split, frame in frames.items():
        assert len(loaded[split]) == len(frame)
        assert loaded[split][features.TEXT_COLUMN].tolist() == frame[
            features.TEXT_COLUMN
        ].tolist()


def test_load_splits_rejects_scenario_in_two_splits(
    tmp_path: Path, frames: dict[str, pd.DataFrame]
) -> None:
    """Сценарий, попавший в две выборки, останавливает подготовку признаков."""
    leak = frames["train"].iloc[[0]].assign(split="validation")
    broken = dict(frames)
    broken["validation"] = pd.concat([frames["validation"], leak], ignore_index=True)
    scenario_id = leak["scenario_id"].iloc[0]

    with pytest.raises(ValueError) as error:
        features.load_splits(_write_splits(broken, tmp_path))

    message = str(error.value)
    assert "выборки пересекаются" in message
    assert "scenario_id" in message
    assert scenario_id in message


def test_load_splits_reports_missing_file(tmp_path: Path) -> None:
    """Отсутствующий файл выборки — это FileNotFoundError с подсказкой про пайплайн."""
    paths = {split: str(tmp_path / f"{split}.csv") for split in SPLIT_NAMES}

    with pytest.raises(FileNotFoundError, match="нет файла"):
        features.load_splits(paths)


def test_cli_writes_manifest(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> None:
    """CLI читает выборки из рабочей директории и пишет манифест по --manifest."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    for split, frame in frames.items():
        frame.to_csv(data_dir / f"incidents_{split}_v1.1.csv", index=False, encoding="utf-8")
    manifest_path = tmp_path / "docs" / "lab6" / "features_manifest.json"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--manifest", str(manifest_path)],
        cwd=tmp_path,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rows"] == {split: len(frame) for split, frame in frames.items()}
    assert manifest["vocabulary_size"] > 0
    assert len(manifest["vocabulary_sha256"]) == 64


def test_cli_returns_error_code_when_splits_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """При отсутствии выборок CLI печатает ошибку, возвращает 1 и не пишет манифест."""
    paths = {split: str(tmp_path / f"{split}.csv") for split in SPLIT_NAMES}
    manifest_path = tmp_path / "features_manifest.json"
    monkeypatch.setattr(features, "SPLIT_FILES", paths)
    monkeypatch.setattr(
        sys, "argv", ["prepare_features.py", "--manifest", str(manifest_path)]
    )

    assert features.main() == 1
    assert not manifest_path.exists()


def test_cli_returns_error_code_on_crossed_splits(
    tmp_path: Path,
    frames: dict[str, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пересечение выборок — тоже отказ с кодом 1, манифест не появляется."""
    leak = frames["train"].iloc[[0]].assign(split="validation")
    broken = dict(frames)
    broken["validation"] = pd.concat([frames["validation"], leak], ignore_index=True)
    paths = _write_splits(broken, tmp_path)
    manifest_path = tmp_path / "features_manifest.json"
    monkeypatch.setattr(features, "SPLIT_FILES", paths)
    monkeypatch.setattr(
        sys, "argv", ["prepare_features.py", "--manifest", str(manifest_path)]
    )

    assert features.main() == 1
    assert not manifest_path.exists()

def test_committed_manifest_matches_the_code(default_splits: dict[str, pd.DataFrame]) -> None:
    """Закоммиченный манифест не должен расходиться с кодом и данными."""
    stored = json.loads(
        (PROJECT_ROOT / "docs" / "lab6" / "features_manifest.json").read_text(encoding="utf-8")
    )
    _, manifest = features.build(default_splits)

    assert manifest["vocabulary_size"] == stored["vocabulary_size"]
    assert manifest["vocabulary_sha256"] == stored["vocabulary_sha256"]
    assert manifest["rows"] == stored["rows"]
    params = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in manifest["params"].items()
    }
    assert params == stored["params"]
