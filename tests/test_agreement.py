"""Согласие разметчиков: разбор меток, каппа Коэна, сверка с набором.

Расчёт опирается на три стороны: метки прохода A, прохода B и набора.
`merge_passes` проставляет `incident_id` обоим листам через ключ, `attach_gold`
добавляет к строкам метки набора, а `agreement` сравнивает любые две стороны.
«Спорный» и пустые ячейки в каппу не входят, а недопустимые значения
останавливают обработку с номером карточки.
"""

from __future__ import annotations

import math
import sys
import warnings

import pandas as pd
import pytest

import agreement


def _sheet(
    cards: list[str],
    severity: list[str],
    service: list[str],
    root_cause: list[str],
    incidents: list[str] | None = None,
) -> pd.DataFrame:
    data = {
        "card": cards,
        "description": [f"описание {card}" for card in cards],
        "severity": severity,
        "service": service,
        "root_cause": root_cause,
    }
    if incidents is not None:
        data["incident_id"] = incidents
    return pd.DataFrame(data)


def _pairs(first: list[str], second: list[str], column: str = "severity") -> pd.DataFrame:
    return pd.DataFrame({f"{column}_label_a": first, f"{column}_label_b": second})


def _key(mapping: dict[str, str], pass_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "card": list(mapping),
            "incident_id": list(mapping.values()),
            "scenario_id": ["SC-TRN-01"] * len(mapping),
            "pass_name": [pass_name] * len(mapping),
        }
    )


def _pairs_from_sheets(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Собирает пару проходов так же, как боевой код: collect → ключ → merge_passes.

    Карточки нумеруются подряд, поэтому первая карточка листа получает I1 и так
    далее; так в результате появляются и метки, и состояния обеих сторон.
    """
    key = pd.concat(
        [
            _key({card: f"I{number}" for number, card in enumerate(first["card"], start=1)}, "a"),
            _key({card: f"I{number}" for number, card in enumerate(second["card"], start=1)}, "b"),
        ],
        ignore_index=True,
    )
    return agreement.merge_passes(
        agreement.collect(first), agreement.collect(second), key
    )


@pytest.fixture
def demo_pairs() -> pd.DataFrame:
    """Демо из постановки: 12 карточек, 10 совпадений, по 6 меток каждого класса."""
    first = ["P1"] * 6 + ["P2"] * 6
    second = ["P1"] * 5 + ["P2"] + ["P1"] + ["P2"] * 5
    return _pairs(first, second)


@pytest.fixture
def merged_pairs() -> pd.DataFrame:
    """Три карточки после сопоставления проходов: у второй во втором проходе пропуск."""
    first = _sheet(
        ["A001", "A002", "A003"],
        ["P1", "P2", "P2"],
        ["billing", "billing", "ims"],
        ["change", "change", "capacity"],
    )
    second = _sheet(
        ["B001", "B002", "B003"],
        ["P1", "", "P3"],
        ["billing", "billing", "ims"],
        ["change", "change", "capacity"],
    )
    return _pairs_from_sheets(first, second)


@pytest.fixture
def gold() -> pd.DataFrame:
    """Метки набора: проход A совпадает с набором в 2 случаях из 3, проход B — в 1 из 2."""
    return pd.DataFrame(
        {
            "incident_id": ["I1", "I2", "I3"],
            "severity": ["P1", "P1", "P2"],
            "service": ["billing", "billing", "ims"],
            "root_cause": ["change", "change", "capacity"],
        }
    )


# --- разбор одной ячейки ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "column", "expected"),
    [
        ("P2", "severity", ("P2", "ok")),
        ("p2", "severity", ("P2", "ok")),
        (" P1 ", "severity", ("P1", "ok")),
        ("Billing", "service", ("billing", "ok")),
        ("External_Dep", "root_cause", ("external_dep", "ok")),
    ],
    ids=["точное", "нижний регистр", "пробелы", "сервис", "причина"],
)
def test_parse_label_accepts_known_values(
    raw: str, column: str, expected: tuple[str, str]
) -> None:
    assert agreement.parse_label(raw, column) == expected


@pytest.mark.parametrize(
    "raw", ["спорный", "Спорно", "не знаю", "Не уверен", "?"], ids=str
)
def test_parse_label_marks_disputed_answers_as_unsure(raw: str) -> None:
    """«Спорный» и его синонимы не становятся меткой и не считаются ошибкой."""
    assert agreement.parse_label(raw, "severity") == (None, "unsure")


@pytest.mark.parametrize("raw", ["", "   ", None], ids=["пусто", "пробелы", "None"])
def test_parse_label_returns_empty_for_blank_cells(raw: object) -> None:
    assert agreement.parse_label(raw, "severity") == (None, "empty")


@pytest.mark.parametrize(
    ("raw", "column"),
    [("P9", "severity"), ("billing", "severity"), ("P1", "service")],
    ids=["P9", "сервис в критичности", "критичность в сервисе"],
)
def test_parse_label_rejects_values_outside_the_catalogue(raw: str, column: str) -> None:
    assert agreement.parse_label(raw, column) == (None, "invalid")


# --- разбор листа -----------------------------------------------------------


def test_collect_adds_label_and_state_columns() -> None:
    frame = _sheet(
        ["A001", "A002", "A003", "A004"],
        ["P1", "p2", "спорный", ""],
        ["billing"] * 4,
        ["change"] * 4,
    )

    result = agreement.collect(frame)

    # pandas хранит пропуск как NaN, поэтому пустые ответы проверяются через isna.
    assert result["severity_label"].isna().tolist() == [False, False, True, True]
    assert result["severity_label"].tolist()[:2] == ["P1", "P2"]
    assert result["severity_state"].tolist() == ["ok", "ok", "unsure", "empty"]


@pytest.mark.parametrize(
    ("column", "value"),
    [("severity", "P9"), ("service", "биллинг"), ("root_cause", "смена")],
    ids=["критичность", "сервис", "причина"],
)
def test_collect_rejects_invalid_value_naming_the_card(column: str, value: str) -> None:
    frame = _sheet(["A001", "A002"], ["P1", "P2"], ["billing"] * 2, ["change"] * 2)
    frame.loc[frame.index[1], column] = value

    with pytest.raises(ValueError, match="A002"):
        agreement.collect(frame)


# --- согласие ---------------------------------------------------------------


def test_agreement_matches_the_demo(demo_pairs: pd.DataFrame) -> None:
    """12 объектов, 10 совпадений, по 6 меток каждого класса: доля 0,8333, каппа 0,6667."""
    stats = agreement.agreement(demo_pairs, "severity")

    assert stats["n"] == 12
    assert stats["agreement"] == pytest.approx(10 / 12, abs=1e-3)
    assert stats["kappa"] == pytest.approx(2 / 3, abs=1e-3)
    assert stats["weighted"] == pytest.approx(2 / 3, abs=1e-3)


def test_agreement_is_one_for_identical_passes() -> None:
    labels = ["P1", "P2", "P3", "P4"]
    stats = agreement.agreement(_pairs(labels, labels), "severity")

    assert stats["n"] == 4
    assert stats["agreement"] == 1.0
    assert stats["kappa"] == pytest.approx(1.0)
    assert stats["weighted"] == pytest.approx(1.0)


def test_unsure_answers_are_excluded_from_comparison() -> None:
    """Спорный ответ не сравнивается, но и не превращается в ошибку."""
    pairs = _pairs(["P1", None, "P2", None], ["P1", "P2", "P2", None])

    stats = agreement.agreement(pairs, "severity")

    assert stats["n"] == 2
    assert stats["agreement"] == 1.0


def test_single_class_gives_nan_kappa() -> None:
    """Каппа на одном классе не определена: доля считается, каппа — NaN без предупреждений."""
    pairs = _pairs(["P1"] * 4, ["P1"] * 4)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = agreement.agreement(pairs, "severity")

    assert stats["n"] == 4
    assert stats["agreement"] == 1.0
    assert math.isnan(stats["kappa"])
    assert math.isnan(stats["weighted"])


def test_no_comparable_pairs_gives_nan_statistics() -> None:
    stats = agreement.agreement(_pairs([None, None], [None, None]), "severity")

    assert stats["n"] == 0
    assert math.isnan(stats["agreement"])
    assert math.isnan(stats["kappa"])


def test_quadratic_weight_treats_neighbouring_errors_as_softer() -> None:
    """Квадратичная каппа различает соседние классы (P1/P2) и далёкие (P1/P4)."""
    first = ["P1", "P1", "P1", "P2", "P2", "P2", "P3", "P3", "P3", "P4", "P4", "P4"]
    near = ["P2", "P1", "P1", "P1", "P2", "P2", "P3", "P3", "P3", "P4", "P4", "P4"]
    far = ["P4", "P1", "P1", "P2", "P2", "P2", "P3", "P3", "P3", "P1", "P4", "P4"]

    near_stats = agreement.agreement(_pairs(first, near), "severity")
    far_stats = agreement.agreement(_pairs(first, far), "severity")

    assert near_stats["agreement"] == far_stats["agreement"] == pytest.approx(10 / 12)
    assert near_stats["kappa"] == pytest.approx(far_stats["kappa"])
    assert near_stats["weighted"] > near_stats["kappa"]
    assert far_stats["weighted"] < far_stats["kappa"]


def test_weighted_kappa_is_kept_for_severity_only() -> None:
    stats = agreement.agreement(
        _pairs(["billing", "ims"], ["billing", "ims"], column="service"), "service"
    )

    assert math.isnan(stats["weighted"])


def test_format_share_uses_russian_decimal_comma() -> None:
    assert agreement.format_share(0.8333) == "0,833"
    assert agreement.format_share(float("nan")) == "—"


def test_agreement_compares_first_and_second_passes(merged_pairs: pd.DataFrame) -> None:
    """По умолчанию сравниваются проходы A и B; пропуск во втором проходе не считается."""
    stats = agreement.agreement(merged_pairs, "severity")

    assert stats["n"] == 2
    assert stats["agreement"] == pytest.approx(1 / 2)


def test_agreement_compares_each_pass_with_the_dataset(
    merged_pairs: pd.DataFrame, gold: pd.DataFrame
) -> None:
    """Стороны a, b и gold: каждый проход сверяется с метками набора отдельно."""
    attached = agreement.attach_gold(merged_pairs, gold)

    by_first = agreement.agreement(attached, "severity", left="a", right="gold")
    by_second = agreement.agreement(attached, "severity", left="b", right="gold")

    assert (by_first["n"], by_second["n"]) == (3, 2)
    assert by_first["agreement"] == pytest.approx(2 / 3)
    assert by_second["agreement"] == pytest.approx(1 / 2)


def test_agreement_rejects_unknown_side(merged_pairs: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="severity_label_gold"):
        agreement.agreement(merged_pairs, "severity", left="a", right="gold")


# --- сопоставление проходов -------------------------------------------------


def test_merge_passes_assigns_incident_id_to_both_sheets(merged_pairs: pd.DataFrame) -> None:
    """Оба листа приходят без incident_id: номер и состояния проставляются по ключу обеих сторон."""
    assert len(merged_pairs) == 3
    assert set(merged_pairs["incident_id"]) == {"I1", "I2", "I3"}
    assert {
        "severity_label_a",
        "severity_label_b",
        "severity_state_a",
        "severity_state_b",
    } <= set(merged_pairs.columns)
    rows = merged_pairs.set_index("incident_id")
    assert (rows.loc["I1", "severity_label_a"], rows.loc["I1", "severity_label_b"]) == (
        "P1",
        "P1",
    )
    assert pd.isna(rows.loc["I2", "severity_label_b"])
    assert rows.loc["I2", "severity_state_b"] == "empty"
    assert rows.loc["I1", "severity_state_b"] == "ok"
    assert (rows.loc["I3", "severity_label_a"], rows.loc["I3", "severity_label_b"]) == (
        "P2",
        "P3",
    )


def test_merge_passes_marks_second_pass_cards_in_the_error() -> None:
    first = agreement.collect(_sheet(["A001"], ["P1"], ["billing"], ["change"]))
    second = agreement.collect(_sheet(["B999"], ["P1"], ["billing"], ["change"]))
    key = pd.concat([_key({"A001": "I1"}, "a"), _key({"B001": "I1"}, "b")])

    with pytest.raises(ValueError) as error:
        agreement.merge_passes(first, second, key)

    message = str(error.value)
    assert "нет в ключе" in message
    assert "B:B999" in message


def test_merge_passes_reports_first_pass_cards_missing_in_the_key() -> None:
    first = agreement.collect(_sheet(["A999"], ["P1"], ["billing"], ["change"]))
    second = agreement.collect(_sheet(["B001"], ["P1"], ["billing"], ["change"]))
    key = pd.concat([_key({"A001": "I1"}, "a"), _key({"B001": "I1"}, "b")])

    with pytest.raises(ValueError, match="A999"):
        agreement.merge_passes(first, second, key)


# --- метки набора -----------------------------------------------------------


def test_attach_gold_adds_dataset_labels(merged_pairs: pd.DataFrame, gold: pd.DataFrame) -> None:
    """Метки набора приходят отдельными колонками, число строк не меняется."""
    attached = agreement.attach_gold(merged_pairs, gold)

    assert len(attached) == len(merged_pairs)
    for column in agreement.ALLOWED:
        assert f"{column}_label_gold" in attached.columns
    labels = attached.set_index("incident_id")["severity_label_gold"]
    assert labels.loc["I1"] == "P1"
    assert labels.loc["I2"] == "P1"
    assert labels.loc["I3"] == "P2"


def test_attach_gold_leaves_unknown_incident_without_label(gold: pd.DataFrame) -> None:
    """Карточки, которой нет в наборе, достаётся NaN, а не выдуманная метка."""
    pairs = pd.DataFrame(
        {
            "card": ["A001"],
            "incident_id": ["I404"],
            "severity_label_a": ["P1"],
            "severity_label_b": ["P1"],
            "service_label_a": ["billing"],
            "service_label_b": ["billing"],
            "root_cause_label_a": ["change"],
            "root_cause_label_b": ["change"],
        }
    )

    attached = agreement.attach_gold(pairs, gold)

    assert len(attached) == 1
    assert pd.isna(attached.loc[0, "severity_label_gold"])


def test_report_compares_each_pass_with_the_dataset(
    merged_pairs: pd.DataFrame, gold: pd.DataFrame
) -> None:
    """Вторая таблица отчёта — совпадение каждого прохода с метками набора.

    Проход A совпадает с набором в 2 случаях из 3 (0,667), проход B — в 1 из 2
    (0,500). Если отчёт сравнивает проходы между собой, в обеих ячейках
    окажется 0,500.
    """
    text = agreement.report(merged_pairs, gold)

    assert "| Критичность | 0,667 | 0,500 |" in text


# --- журнал расхождений -----------------------------------------------------


@pytest.fixture
def pairs_with_gold() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Четыре карточки: полное совпадение, расхождение с набором и расхождение проходов."""
    first = _sheet(
        ["A001", "A002", "A003", "A004"],
        ["P1", "P2", "P2", "P3"],
        ["billing", "billing", "ims", "ims"],
        ["change", "capacity", "software", "network"],
    )
    second = _sheet(
        ["B001", "B002", "B003", "B004"],
        ["P1", "P2", "P3", "P3"],
        ["billing", "billing", "ims", "transport"],
        ["change", "capacity", "software", "hardware"],
    )
    gold = pd.DataFrame(
        {
            "incident_id": ["I1", "I2", "I3", "I4"],
            "severity": ["P1", "P3", "P2", "P3"],
            "service": ["billing", "billing", "ims", "ims"],
            "root_cause": ["change", "capacity", "software", "network"],
        }
    )
    return _pairs_from_sheets(first, second), gold


def test_discrepancies_logs_only_divergent_rows(
    pairs_with_gold: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    pairs, gold = pairs_with_gold

    log = agreement.discrepancies(pairs, gold)

    found = set(zip(log["incident_id"], log["label"]))
    assert found == {
        ("I2", "Критичность"),
        ("I3", "Критичность"),
        ("I4", "Сервис"),
        ("I4", "Категория причины"),
    }


def test_discrepancies_never_lists_a_full_match(
    pairs_with_gold: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    pairs, gold = pairs_with_gold

    log = agreement.discrepancies(pairs, gold)

    assert "I1" not in set(log["incident_id"])


def test_discrepancies_keeps_both_passes_and_the_dataset(
    pairs_with_gold: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    pairs, gold = pairs_with_gold

    log = agreement.discrepancies(pairs, gold)
    row = log.set_index(["incident_id", "label"]).loc[("I3", "Критичность")]

    assert row["card"] == "A003"
    assert (row["first"], row["second"], row["dataset"]) == ("P2", "P3", "P2")


def test_discrepancies_skips_cards_without_an_answer() -> None:
    """Пустая ячейка означает, что метку не размечали: это не расхождение."""
    first = _sheet(["A001"], ["P2"], ["billing"], ["change"])
    second = _sheet(["B001"], [""], ["billing"], ["change"])
    gold = pd.DataFrame(
        {
            "incident_id": ["I1"],
            "severity": ["P1"],
            "service": ["billing"],
            "root_cause": ["change"],
        }
    )

    log = agreement.discrepancies(_pairs_from_sheets(first, second), gold)

    assert log.empty


def test_discrepancies_skips_cards_empty_in_both_passes() -> None:
    """Если метку не размечали ни в одном проходе, журналу нечего сравнивать."""
    first = _sheet(["A001"], [""], ["billing"], ["change"])
    second = _sheet(["B001"], [""], ["billing"], ["change"])
    gold = pd.DataFrame(
        {
            "incident_id": ["I1"],
            "severity": ["P1"],
            "service": ["billing"],
            "root_cause": ["change"],
        }
    )

    log = agreement.discrepancies(_pairs_from_sheets(first, second), gold)

    assert log.empty


def test_discrepancies_logs_disputed_answers() -> None:
    """«Спорный» — это ответ, поэтому он попадает в журнал с пустой меткой."""
    first = _sheet(["A001"], ["P2"], ["billing"], ["change"])
    second = _sheet(["B001"], ["спорный"], ["billing"], ["change"])
    gold = pd.DataFrame(
        {
            "incident_id": ["I1"],
            "severity": ["P1"],
            "service": ["billing"],
            "root_cause": ["change"],
        }
    )

    log = agreement.discrepancies(_pairs_from_sheets(first, second), gold)

    assert len(log) == 1
    row = log.iloc[0]
    assert (row["incident_id"], row["label"]) == ("I1", "Критичность")
    assert row["first"] == "P2"
    assert pd.isna(row["second"])
    assert row["dataset"] == "P1"


# --- CLI --------------------------------------------------------------------


@pytest.fixture
def annotation_files(tmp_path: Path) -> tuple[Path, Path]:
    """Листы, ключ и метки набора в том виде, в каком их читает main()."""
    labels = tmp_path / "annotation"
    labels.mkdir()
    first = _sheet(
        ["A001", "A002", "A003"],
        ["P1", "P2", "P2"],
        ["billing", "billing", "ims"],
        ["change", "change", "capacity"],
    )
    second = _sheet(
        ["B001", "B002", "B003"],
        ["P1", "", "P3"],
        ["billing", "billing", "ims"],
        ["change", "change", "capacity"],
    )
    key = pd.concat(
        [
            _key({"A001": "I1", "A002": "I2", "A003": "I3"}, "a"),
            _key({"B001": "I1", "B002": "I2", "B003": "I3"}, "b"),
        ],
        ignore_index=True,
    )
    gold = pd.DataFrame(
        {
            "incident_id": ["I1", "I2", "I3"],
            "severity": ["P1", "P1", "P2"],
            "service": ["billing", "billing", "ims"],
            "root_cause": ["change", "change", "capacity"],
        }
    )
    for name, frame in (("pass_a.csv", first), ("pass_b.csv", second), ("key.csv", key)):
        frame.to_csv(labels / name, index=False, encoding="utf-8")
    gold_path = tmp_path / "validation.csv"
    gold.to_csv(gold_path, index=False, encoding="utf-8")
    return labels, gold_path


def test_cli_writes_the_full_report(
    annotation_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Сквозной путь main(): листы без incident_id читаются, ключ связывает проходы."""
    labels, gold_path = annotation_files
    monkeypatch.setattr(
        sys,
        "argv",
        ["agreement.py", "--labels", str(labels), "--data", str(gold_path)],
    )

    assert agreement.main() == 0

    out = capsys.readouterr().out
    assert "Согласие первого и второго прохода" in out
    assert "| Критичность | 0,667 | 0,500 |" in out
    # I2 во втором проходе не размечен: пустая ячейка в журнал не идёт.
    assert "Расхождений в журнале: 1" in out
    assert "A003: Критичность" in out
    assert "A002" not in out


def test_cli_stops_on_an_invalid_label(
    annotation_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels, gold_path = annotation_files
    sheet = pd.read_csv(
        labels / "pass_a.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    sheet.loc[0, "severity"] = "P9"
    sheet.to_csv(labels / "pass_a.csv", index=False, encoding="utf-8-sig")
    monkeypatch.setattr(
        sys,
        "argv",
        ["agreement.py", "--labels", str(labels), "--data", str(gold_path)],
    )

    assert agreement.main() == 1

    assert "A001" in capsys.readouterr().out
