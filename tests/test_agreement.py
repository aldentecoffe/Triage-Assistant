"""Согласие разметчиков: две таблицы, каппа и журнал расхождений.

Первая таблица сравнивает проходы между собой только на повторно размеченных
карточках. Вторая сверяет каждый проход с метками набора: первый проход берётся
целиком, поэтому `combine` шире `merge_passes`. Тесты держат это различие,
квадратичную каппу в выводе и правила журнала: пустая ячейка сама по себе
расхождением не считается, «спорный» и противоречие набору — считаются.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

import agreement
from incident_taxonomy import ROOT_CAUSES, SERVICES, SEVERITIES


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


def _combined_case(first_value: str, second_value: str, gold_value: str) -> pd.DataFrame:
    """Одна карточка и её повтор: значения метки критичности задаются явно."""
    first = _sheet(["A001"], [first_value], ["billing"], ["change"])
    second = _sheet(["B001"], [second_value], ["billing"], ["change"])
    key = pd.concat(
        [_key({"A001": "I1"}, "a"), _key({"B001": "I1"}, "b")], ignore_index=True
    )
    gold = pd.DataFrame(
        {
            "incident_id": ["I1"],
            "severity": [gold_value],
            "service": ["billing"],
            "root_cause": ["change"],
        }
    )
    return agreement.combine(
        agreement.collect(first), agreement.collect(second), key, gold
    )


def _row_after(text: str, heading: str, name: str) -> list[str]:
    """Строка таблицы из нужного раздела отчёта, разобранная по ячейкам."""
    block = text.split(heading, 1)[1]
    for line in block.splitlines():
        if line.startswith(f"| {name} |"):
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"в разделе {heading!r} нет строки {name!r}")


def _cli_error(capsys: pytest.CaptureFixture[str]) -> str:
    """Проверяет, что CLI напечатал «Ошибка:» и не оставил трассировку."""
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "Ошибка:" in text
    assert "Traceback" not in text
    return text


# --- фикстуры ---------------------------------------------------------------


@pytest.fixture
def pilot_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Пилот в миниатюре: первый проход шире второго, карточку I3 не повторяли.

    Проход A размечен целиком: P1, P2, P2. Проход B повторяет I1 и I2, причём
    по сервису в I1 он расходится с A. Набор по критичности расходится с A
    в карточке I3, которой во втором проходе нет.
    """
    first = _sheet(
        ["A001", "A002", "A003"],
        ["P1", "P2", "P2"],
        ["billing", "billing", "ims"],
        ["change", "change", "capacity"],
    )
    second = _sheet(
        ["B001", "B002"],
        ["P1", "P2"],
        ["ims", "billing"],
        ["change", "change"],
    )
    key = pd.concat(
        [
            _key({"A001": "I1", "A002": "I2", "A003": "I3"}, "a"),
            _key({"B001": "I1", "B002": "I2"}, "b"),
        ],
        ignore_index=True,
    )
    gold = pd.DataFrame(
        {
            "incident_id": ["I1", "I2", "I3"],
            "severity": ["P1", "P2", "P1"],
            "service": ["billing", "billing", "ims"],
            "root_cause": ["change", "change", "capacity"],
        }
    )
    return first, second, key, gold


@pytest.fixture
def merged(pilot_frames) -> pd.DataFrame:
    """Только карточки, размеченные дважды: I1 и I2."""
    first, second, key, _ = pilot_frames
    return agreement.merge_passes(
        agreement.collect(first), agreement.collect(second), key
    )


@pytest.fixture
def combined(pilot_frames) -> pd.DataFrame:
    """Весь первый проход: I1, I2 и I3 с пустым вторым проходом."""
    first, second, key, gold = pilot_frames
    return agreement.combine(
        agreement.collect(first), agreement.collect(second), key, gold
    )


@pytest.fixture
def annotation_files(
    tmp_path: Path, pilot_frames
) -> tuple[Path, Path]:
    """Листы, ключ и метки набора в том виде, в каком их читает main()."""
    first, second, key, gold = pilot_frames
    labels = tmp_path / "annotation"
    labels.mkdir()
    for name, frame in (("pass_a.csv", first), ("pass_b.csv", second), ("key.csv", key)):
        frame.to_csv(labels / name, index=False, encoding="utf-8")
    gold_path = tmp_path / "validation.csv"
    gold.to_csv(gold_path, index=False, encoding="utf-8")
    return labels, gold_path


def _run_cli(monkeypatch: pytest.MonkeyPatch, labels: Path, gold: Path) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        ["agreement.py", "--labels", str(labels), "--data", str(gold)],
    )
    return agreement.main()


@pytest.fixture
def demo_pairs() -> pd.DataFrame:
    """Демо из постановки: 12 карточек, 10 совпадений, по 6 меток каждого класса."""
    first = ["P1"] * 6 + ["P2"] * 6
    second = ["P1"] * 5 + ["P2"] + ["P1"] + ["P2"] * 5
    return _pairs(first, second)


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


def test_agreement_compares_first_and_second_passes(merged: pd.DataFrame) -> None:
    """По умолчанию сравниваются проходы A и B на повторно размеченных карточках."""
    stats = agreement.agreement(merged, "severity")

    assert stats["n"] == 2
    assert stats["agreement"] == 1.0


def test_agreement_compares_each_pass_with_the_dataset(combined: pd.DataFrame) -> None:
    """У прохода A в сверке три карточки, у прохода B — только повторённые две."""
    by_first = agreement.agreement(combined, "severity", left="a", right="gold")
    by_second = agreement.agreement(combined, "severity", left="b", right="gold")

    assert (by_first["n"], by_second["n"]) == (3, 2)
    assert by_first["agreement"] == pytest.approx(2 / 3)
    assert by_second["agreement"] == 1.0


def test_agreement_rejects_unknown_side(combined: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="severity_label_c"):
        agreement.agreement(combined, "severity", left="a", right="c")


def test_format_share_uses_russian_decimal_comma() -> None:
    assert agreement.format_share(0.8333) == "0,833"
    assert agreement.format_share(float("nan")) == "—"


def test_format_value_replaces_empty_labels_with_a_dash() -> None:
    assert agreement.format_value(float("nan")) == "—"
    assert agreement.format_value(None) == "—"
    assert agreement.format_value("P1") == "P1"


# --- подготовка стороны и сопоставление -------------------------------------


def test_prepare_side_maps_cards_and_renames_columns(pilot_frames) -> None:
    first, _, key, _ = pilot_frames

    prepared = agreement.prepare_side(agreement.collect(first), key, "a")

    assert prepared.set_index("card").loc["A002", "incident_id"] == "I2"
    assert {"severity_label_a", "severity_state_a"} <= set(prepared.columns)
    assert "severity_label" not in prepared.columns


def test_merge_passes_keeps_only_repeated_cards(merged: pd.DataFrame) -> None:
    """Внутреннее соединение оставляет карточки, размеченные дважды."""
    assert len(merged) == 2
    assert set(merged["incident_id"]) == {"I1", "I2"}
    assert {
        "severity_label_a",
        "severity_label_b",
        "severity_state_a",
        "severity_state_b",
    } <= set(merged.columns)


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


# --- метки набора и полный проход -------------------------------------------


def test_attach_gold_adds_dataset_labels(merged: pd.DataFrame, pilot_frames) -> None:
    """Метки набора приходят отдельными колонками, число строк не меняется."""
    _, _, _, gold = pilot_frames

    attached = agreement.attach_gold(merged, gold)

    assert len(attached) == len(merged)
    for column in agreement.ALLOWED:
        assert f"{column}_label_gold" in attached.columns
    labels = attached.set_index("incident_id")["severity_label_gold"]
    assert labels.loc["I1"] == "P1"
    assert labels.loc["I2"] == "P2"


def test_attach_gold_leaves_unknown_incident_without_label(pilot_frames) -> None:
    """Карточке, которой нет в наборе, достаётся NaN, а не выдуманная метка."""
    _, _, _, gold = pilot_frames
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


def test_combine_keeps_the_whole_first_pass(
    combined: pd.DataFrame, merged: pd.DataFrame
) -> None:
    """Сверка с набором идёт по всем карточкам первого прохода, а не по повторённым."""
    assert len(combined) == 3
    assert len(merged) == 2
    assert set(combined["incident_id"]) == {"I1", "I2", "I3"}
    assert {
        "severity_label_gold",
        "service_label_gold",
        "root_cause_label_gold",
    } <= set(combined.columns)


def test_combine_marks_second_pass_empty_where_not_repeated(
    combined: pd.DataFrame,
) -> None:
    rows = combined.set_index("incident_id")

    assert pd.isna(rows.loc["I3", "severity_label_b"])
    assert rows.loc["I3", "severity_state_b"] == "empty"
    assert rows.loc["I1", "severity_state_b"] == "ok"
    assert combined["severity_state_b"].notna().all()


def test_combine_covers_the_full_pilot() -> None:
    """Боевой масштаб: 120 карточек первого прохода и 30 повторных."""
    count, repeat = 120, 30
    severities = [SEVERITIES[index % len(SEVERITIES)] for index in range(count)]
    services = [list(SERVICES)[index % len(SERVICES)] for index in range(count)]
    causes = [list(ROOT_CAUSES)[index % len(ROOT_CAUSES)] for index in range(count)]
    first = _sheet(
        [f"A{number:03d}" for number in range(1, count + 1)],
        severities,
        services,
        causes,
    )
    second = _sheet(
        [f"B{number:03d}" for number in range(1, repeat + 1)],
        severities[:repeat],
        services[:repeat],
        causes[:repeat],
    )
    key = pd.concat(
        [
            _key({f"A{number:03d}": f"I{number}" for number in range(1, count + 1)}, "a"),
            _key({f"B{number:03d}": f"I{number}" for number in range(1, repeat + 1)}, "b"),
        ],
        ignore_index=True,
    )
    gold = pd.DataFrame(
        {
            "incident_id": [f"I{number}" for number in range(1, count + 1)],
            "severity": severities,
            "service": services,
            "root_cause": causes,
        }
    )

    prepared_first = agreement.collect(first)
    prepared_second = agreement.collect(second)
    combined = agreement.combine(prepared_first, prepared_second, key, gold)

    assert len(combined) == count
    assert len(agreement.merge_passes(prepared_first, prepared_second, key)) == repeat
    assert int(combined["severity_state_b"].eq("empty").sum()) == count - repeat
    assert int(combined["severity_label_b"].notna().sum()) == repeat
    by_first = agreement.agreement(combined, "severity", "a", "gold")
    by_second = agreement.agreement(combined, "severity", "b", "gold")
    assert (by_first["n"], by_second["n"]) == (count, repeat)


# --- отчёт ------------------------------------------------------------------


def test_report_compares_each_pass_with_the_dataset(combined: pd.DataFrame) -> None:
    """Вторая таблица: у прохода A три сравнения, у прохода B — два."""
    text = agreement.report(combined)

    assert (
        "| Метка | Проход A: сравнено | Проход A: доля | "
        "Проход B: сравнено | Проход B: доля |" in text
    )
    row = _row_after(text, "### Сверка с метками набора", "Критичность")

    assert row == ["Критичность", "3", "0,667", "2", "1,000"]


def test_report_prints_quadratic_kappa_for_severity_only(
    combined: pd.DataFrame,
) -> None:
    """Квадратичная каппа заполнена только для критичности, у сервиса и причины — «—»."""
    text = agreement.report(combined)

    assert "Квадратичная каппа" in text
    severity = _row_after(text, "### Согласие первого и второго прохода", "Критичность")
    service = _row_after(text, "### Согласие первого и второго прохода", "Сервис")
    cause = _row_after(text, "### Согласие первого и второго прохода", "Категория причины")

    assert severity[5] == "1,000"
    assert service[5] == "—"
    assert cause[5] == "—"


# --- журнал расхождений -----------------------------------------------------


def test_discrepancies_logs_divergence_and_dataset_conflicts(combined: pd.DataFrame) -> None:
    """В журнал попадают расхождение проходов и карточка, которой не было в повторе."""
    log = agreement.discrepancies(combined)

    found = set(zip(log["incident_id"], log["label"]))
    assert found == {("I1", "Сервис"), ("I3", "Критичность")}


def test_discrepancies_skips_an_empty_answer() -> None:
    """Пустая ячейка второго прохода сама по себе расхождения не создаёт."""
    log = agreement.discrepancies(_combined_case("P2", "", "P2"))

    assert log.empty


def test_discrepancies_skips_unrepeated_cards_that_match_the_dataset() -> None:
    """Карточка без повтора не попадает в журнал, если первый проход совпал с набором."""
    first = agreement.collect(
        _sheet(["A001", "A002"], ["P1", "P2"], ["billing"] * 2, ["change"] * 2)
    )
    second = agreement.collect(_sheet(["B001"], ["P1"], ["billing"], ["change"]))
    key = pd.concat(
        [
            _key({"A001": "I1", "A002": "I2"}, "a"),
            _key({"B001": "I1"}, "b"),
        ],
        ignore_index=True,
    )
    gold = pd.DataFrame(
        {
            "incident_id": ["I1", "I2"],
            "severity": ["P1", "P2"],
            "service": ["billing", "billing"],
            "root_cause": ["change", "change"],
        }
    )

    combined = agreement.combine(first, second, key, gold)

    assert agreement.discrepancies(combined).empty


def test_discrepancies_skips_cards_empty_in_both_passes() -> None:
    """Если метку не размечали ни в одном проходе, журналу нечего сравнивать."""
    log = agreement.discrepancies(_combined_case("", "", "P1"))

    assert log.empty


def test_discrepancies_logs_disputed_answers() -> None:
    """«Спорный» — это ответ, поэтому он попадает в журнал с пустой меткой."""
    log = agreement.discrepancies(_combined_case("P2", "спорный", "P2"))

    assert len(log) == 1
    row = log.iloc[0]
    assert (row["first"], row["dataset"]) == ("P2", "P2")
    assert pd.isna(row["second"])


def test_discrepancies_logs_passes_that_differ() -> None:
    log = agreement.discrepancies(_combined_case("P2", "P3", "P2"))

    assert len(log) == 1
    assert (log.iloc[0]["first"], log.iloc[0]["second"]) == ("P2", "P3")


def test_discrepancies_logs_agreement_against_the_dataset() -> None:
    """Оба прохода совпали между собой, но расходятся с набором."""
    log = agreement.discrepancies(_combined_case("P2", "P2", "P1"))

    assert len(log) == 1
    assert (log.iloc[0]["first"], log.iloc[0]["dataset"]) == ("P2", "P1")


# --- ключ -------------------------------------------------------------------


def test_read_key_returns_the_mapping(tmp_path: Path) -> None:
    path = tmp_path / "key.csv"
    _key({"A001": "I1"}, "a").to_csv(path, index=False, encoding="utf-8")

    loaded = agreement.read_key(path)

    assert {"card", "incident_id", "pass_name"} <= set(loaded.columns)
    assert len(loaded) == 1


def test_read_key_rejects_empty_key(tmp_path: Path) -> None:
    path = tmp_path / "key.csv"
    pd.DataFrame(columns=["card", "incident_id", "pass_name"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="пуст"):
        agreement.read_key(path)


def test_read_key_requires_all_columns(tmp_path: Path) -> None:
    path = tmp_path / "key.csv"
    _key({"A001": "I1"}, "a").drop(columns=["pass_name"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="pass_name"):
        agreement.read_key(path)


def test_read_key_rejects_duplicate_card_numbers(tmp_path: Path) -> None:
    path = tmp_path / "key.csv"
    duplicated = pd.concat(
        [_key({"A001": "I1"}, "a"), _key({"A001": "I1"}, "a")], ignore_index=True
    )
    duplicated.to_csv(path, index=False)

    with pytest.raises(ValueError, match="повторяются"):
        agreement.read_key(path)


# --- CLI --------------------------------------------------------------------


def test_cli_writes_the_full_report(
    annotation_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels, gold_path = annotation_files

    assert _run_cli(monkeypatch, labels, gold_path) == 0

    out = capsys.readouterr().out
    assert "| Критичность | 3 | 0,667 | 2 | 1,000 |" in out
    assert "Расхождений в журнале: 2" in out
    assert "A001: Сервис" in out
    assert "A003: Критичность" in out
    assert "nan" not in out
    assert "B —" in out


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

    assert _run_cli(monkeypatch, labels, gold_path) == 1

    assert "A001" in _cli_error(capsys)


def test_cli_stops_on_a_broken_key(
    annotation_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels, gold_path = annotation_files
    key = pd.read_csv(
        labels / "key.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    key.drop(columns=["pass_name"]).to_csv(labels / "key.csv", index=False)

    assert _run_cli(monkeypatch, labels, gold_path) == 1

    assert "pass_name" in _cli_error(capsys)


def test_cli_stops_on_a_duplicate_key(
    annotation_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels, gold_path = annotation_files
    key = pd.read_csv(
        labels / "key.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    pd.concat([key, key.iloc[[0]]], ignore_index=True).to_csv(
        labels / "key.csv", index=False
    )

    assert _run_cli(monkeypatch, labels, gold_path) == 1

    assert "повторяются" in _cli_error(capsys)


def test_cli_stops_when_data_lacks_incident_id(
    annotation_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels, gold_path = annotation_files
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    gold.drop(columns=["incident_id"]).to_csv(gold_path, index=False)

    assert _run_cli(monkeypatch, labels, gold_path) == 1

    assert "incident_id" in _cli_error(capsys)

def test_combine_keeps_every_card_when_second_pass_is_empty(pilot_frames) -> None:
    """Пустой второй проход: все карточки A на месте, состояния B — empty, без предупреждений."""
    first, _, key, gold = pilot_frames
    empty_second = _sheet([], [], [], [])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        combined = agreement.combine(
            agreement.collect(first), agreement.collect(empty_second), key, gold
        )

    assert len(combined) == len(first)
    assert set(combined["severity_state_b"]) == {"empty"}
    assert combined["severity_label_b"].isna().all()


def test_combine_rejects_second_pass_card_missing_in_the_first(pilot_frames) -> None:
    """Ключ, ведущий во второй проход на чужой incident_id, не должен молча терять карточку."""
    first, second, key, gold = pilot_frames
    broken = key.copy()
    broken.loc[broken["pass_name"] == "b", "incident_id"] = "I999"

    with pytest.raises(ValueError, match="во втором листе"):
        agreement.combine(agreement.collect(first), agreement.collect(second), broken, gold)


def test_read_key_rejects_unknown_pass_name(tmp_path: Path) -> None:
    path = tmp_path / "key.csv"
    frame = _key({"A001": "I1"}, "a").assign(pass_name="c")
    frame.to_csv(path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="неизвестные проходы"):
        agreement.read_key(path)
