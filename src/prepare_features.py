"""Подготовка признаков без утечки из test.

Порядок шага из ЛР 6: сначала проверка, что сценарии и события не пересекают
выборки, затем словарь и IDF считаются только по train, и уже неизменные веса
применяются к validation и test. Словарь и параметры фиксируются до выбора
модели: если после первого обучения поменять min_df или n-граммы, сравнение
моделей перестаёт быть честным.

Запуск из корня репозитория:
    python src/prepare_features.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from split_dataset import find_violations

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from scipy.sparse import csr_matrix

FEATURE_VERSION = "1.0"
TEXT_COLUMN = "description"
SPLIT_FILES = {
    "train": "data/processed/incidents_train_v1.1.csv",
    "validation": "data/processed/incidents_validation_v1.1.csv",
    "test": "data/processed/incidents_test_v1.1.csv",
}

# Параметры зафиксированы до первого обучения модели и не подбираются по test.
VECTORIZER_PARAMS = {
    "analyzer": "word",
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 20000,
    "sublinear_tf": True,
}


def load_splits(paths: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Читает три выборки и проверяет, что они разведены по группам."""
    paths = paths or SPLIT_FILES
    frames = {}
    for split, path in paths.items():
        try:
            frames[split] = pd.read_csv(path, dtype=str, keep_default_na=False)
        except OSError as error:
            raise FileNotFoundError(f"нет файла {path}: сначала пройдите пайплайн") from error

    combined = pd.concat(frames.values(), ignore_index=True)
    violations = find_violations(combined)
    if violations:
        raise ValueError("выборки пересекаются: " + "; ".join(violations))
    return frames


def fit_vectorizer(train: pd.DataFrame) -> TfidfVectorizer:
    """Считает словарь и IDF только по train."""
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    vectorizer.fit(train[TEXT_COLUMN].astype(str))
    return vectorizer


def transform(vectorizer: TfidfVectorizer, frame: pd.DataFrame) -> "csr_matrix":
    """Применяет неизменные веса: новые слова в других выборках игнорируются."""
    return vectorizer.transform(frame[TEXT_COLUMN].astype(str))


def vocabulary_hash(vectorizer: TfidfVectorizer) -> str:
    """Отпечаток словаря: по нему видно, что состав признаков не поменялся."""
    terms = sorted(vectorizer.get_feature_names_out())
    return hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()


def build(frames: dict[str, pd.DataFrame]) -> tuple[TfidfVectorizer, dict[str, object]]:
    """Обучает векторизатор на train и возвращает его вместе с манифестом.

    Матрицы считаются внутри (`transform` для каждой выборки) и в манифест
    попадают только их размеры: сами разреженные матрицы наружу не отдаются,
    чтобы вызывающий код не мог подменить веса.
    """
    vectorizer = fit_vectorizer(frames["train"])
    matrices = {split: transform(vectorizer, frames[split]) for split in frames}
    manifest = {
        "feature_version": FEATURE_VERSION,
        "text_column": TEXT_COLUMN,
        "params": {key: value for key, value in VECTORIZER_PARAMS.items()},
        "vocabulary_size": len(vectorizer.vocabulary_),
        "vocabulary_sha256": vocabulary_hash(vectorizer),
        "rows": {split: int(matrix.shape[0]) for split, matrix in matrices.items()},
        "features": int(matrices["train"].shape[1]),
    }
    return vectorizer, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="docs/lab6/features_manifest.json")
    args = parser.parse_args()

    try:
        frames = load_splits()
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    vectorizer, manifest = build(frames)
    print(f"словарь: {manifest['vocabulary_size']} признаков")
    print(f"отпечаток словаря: {manifest['vocabulary_sha256']}")
    for split, rows in manifest["rows"].items():
        print(f"{split}: {rows} строк")

    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"манифест: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
