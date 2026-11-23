"""Проверки качества EDA-модуля лабораторной 5.

Тесты покрывают вспомогательные расчёты (`template_key`, `cramers_v`,
`token_coverage`, `proxy_table`) и детекторы `checks_*`: корректный набор
каждая проверка обязана пропускать, а внесённую порчу — ловить. Данные строятся генератором,
поэтому файлы в `data/` и сеть тестам не нужны.

Набор в 140 записей: при таком объёме каждый сценарий получает несколько
строк, все семь сервисов попадают в каждый split, а прогон занимает
миллисекунды.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eda_incidents import (  # noqa: E402
    AUDIT_FIELDS,
    DATA_PATHS,
    PROXY_WARN_LEVEL,
    SPLIT_ORDER,
    checks_allowed_values,
    checks_cascades,
    checks_class_support,
    checks_cleaning_log,
    checks_coverage,
    checks_duplicates,
    checks_events,
    checks_keys,
    checks_proxy,
    checks_required,
    checks_schema,
    checks_split_crossings,
    checks_split_files,
    checks_templates,
    checks_text_length,
    checks_time,
    checks_title,
    checks_volume,
    cramers_v,
    display_path,
    load_frames,
    main,
    proxy_table,
    ru,
    run_checks,
    template_groups,
    template_key,
    token_coverage,
)
from generate_incidents import build_rows  # noqa: E402
from incident_taxonomy import CASCADES, SEVERITIES  # noqa: E402
from prepare_dataset import clean  # noqa: E402
from split_dataset import assign  # noqa: E402

SMALL_TOTAL = 140  # 84 записи в train и по 28 в validation и test
MEDIUM_TOTAL = 600  # набор для run_checks: объём меньше 6 000, но все выборки полны
INDEPENDENCE_SEED = 42


# --- данные ----------------------------------------------------------------


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """Сырой набор с контрольными дефектами."""
    return pd.DataFrame(build_rows(total=SMALL_TOTAL))


@pytest.fixture(scope="module")
def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    """Очищенный набор без дефектов: вход для проверок качества."""
    return clean(raw)[0]


@pytest.fixture(scope="module")
def cleaning_log(raw: pd.DataFrame) -> pd.DataFrame:
    """Журнал исключений с колонкой stage в целочисленном виде."""
    return clean(raw)[1]


@pytest.fixture(scope="module")
def assigned(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Очищенный набор с split по каталогу сценариев."""
    return assign(cleaned)


@pytest.fixture(scope="module")
def medium_frames() -> dict[str, pd.DataFrame]:
    """Шесть наборов в том виде, в каком их получает run_checks."""
    raw = pd.DataFrame(build_rows(total=MEDIUM_TOTAL))
    cleaned, log = clean(raw)
    frame = assign(cleaned)
    frames = {"raw": raw, "clean": cleaned, "log": log}
    for split in SPLIT_ORDER:
        frames[split] = frame[frame["split"] == split].reset_index(drop=True)
    return frames


def _report(rows: list[dict[str, str]]) -> dict[str, str]:
    """Единственная строка отчёта одной проверки."""
    assert len(rows) == 1, "проверка должна возвращать ровно одну строку"
    return rows[0]


def _other_split(split: str) -> str:
    """Любая выборка, отличная от текущей."""
    return next(name for name in SPLIT_ORDER if name != split)


def _plain(text: str) -> str:
    """Отчёт печатает десятичную запятую; для сверки чисел приводим её к точке."""
    return text.replace(",", ".")


# --- template_key ----------------------------------------------------------


def test_template_key_normalises_ci_and_numbers() -> None:
    # Две карточки одного шаблона: разные CI, значения метрики и масштаб последствий.
    first = (
        "Падает BGP-сессия с пирингом. route_withdrawals = 120 шт/мин, порог 50 шт/мин. "
        "Около 400 префиксов ушло в резерв. Объект CI-100234."
    )
    second = (
        "Падает BGP-сессия с пирингом. route_withdrawals = 300 шт/мин, порог 50 шт/мин. "
        "Около 900 префиксов ушло в резерв. Объект CI-999999."
    )

    assert template_key(first) == template_key(second)


def test_template_key_normalises_host_name() -> None:
    # Вторая формулировка генератора: узел стоит внутри текста и меняется от карточки к карточке.
    first = (
        "Падает BGP-сессия с пирингом на host-msk123. route_withdrawals держится "
        "155 шт/мин при пороге 50 шт/мин. Около 400 префиксов ушло в резерв. Объект CI-100234."
    )
    second = (
        "Падает BGP-сессия с пирингом на host-spb007. route_withdrawals держится "
        "210 шт/мин при пороге 50 шт/мин. Около 400 префиксов ушло в резерв. Объект CI-100234."
    )

    assert template_key(first) == template_key(second)


def test_template_key_separates_different_symptoms() -> None:
    bgp = "Падает BGP-сессия с пирингом. Около 400 префиксов ушло в резерв. Объект CI-100234."
    fib = "Таблица маршрутизации упирается в лимит FIB на host-nsk017. Объект CI-100235."
    # Тот же симптом, но другой язык описания — тоже другой шаблон.
    bgp_en = "BGP session with the upstream peer keeps flapping. Item CI-100234."

    assert template_key(bgp) != template_key(fib)
    assert template_key(bgp) != template_key(bgp_en)


def test_template_key_is_stable_short_hash() -> None:
    text = "Падает BGP-сессия с пирингом. Объект CI-100234."

    assert template_key(text) == template_key(text)
    assert len(template_key(text)) == 16
    assert set(template_key(text)) <= set("0123456789abcdef")
    # Пробельное описание не даёт исключения: получается ключ пустой строки.
    assert template_key("") == template_key("  \n\t ")


# --- cramers_v -------------------------------------------------------------


def test_cramers_v_is_one_for_functional_dependence() -> None:
    """Одно поле однозначно определяется другим: связь максимальна."""
    left = pd.Series(["a", "a", "b", "b", "c", "c"])
    right = pd.Series(["x", "x", "y", "y", "z", "z"])

    assert cramers_v(left, right) == pytest.approx(1.0)


def test_cramers_v_is_partial_for_noisy_dependence() -> None:
    """Связь есть, но не функциональная: значение между нулём и единицей."""
    left = pd.Series(["a", "a", "a", "b", "b", "b"])
    right = pd.Series(["x", "x", "y", "x", "x", "x"])

    value = cramers_v(left, right)

    assert value == pytest.approx(0.447, abs=0.01)
    assert 0.0 < value < 1.0


def test_cramers_v_is_near_zero_for_independent_fields() -> None:
    rng = np.random.default_rng(INDEPENDENCE_SEED)
    left = pd.Series(rng.choice(["a", "b", "c", "d"], size=5000))
    right = pd.Series(rng.choice(["w", "x", "y", "z"], size=5000))

    value = cramers_v(left, right)

    assert 0.0 <= value < 0.05


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (pd.Series([], dtype=object), pd.Series([], dtype=object)),
        (pd.Series(["a", "a", "a"]), pd.Series(["x", "y", "x"])),
        (pd.Series(["x", "y", "x"]), pd.Series(["a", "a", "a"])),
    ],
    ids=["пустые", "левое константное", "правое константное"],
)
def test_cramers_v_returns_zero_on_degenerate_input(left: pd.Series, right: pd.Series) -> None:
    """Вырожденный вход не должен ронять расчёт: у него просто нет связи."""
    assert cramers_v(left, right) == 0.0


# --- token_coverage --------------------------------------------------------


def test_token_coverage_counts_tokens_and_documents() -> None:
    reference = {"bgp", "сессия", "упала"}
    texts = pd.Series(["BGP-сессия упала", "bgp упала снова 12"])

    share, documents = token_coverage(reference, texts)

    # Токенов известно 5 из 6 (цифры регулярка отбрасывает).
    assert share == pytest.approx(5 / 6)
    # Порог 80 % проходит только первая карточка: 3/3 против 2/3.
    assert documents == pytest.approx(0.5)


def test_token_coverage_threshold_is_inclusive() -> None:
    reference = {"alpha", "beta", "gamma", "delta"}

    share, documents = token_coverage(reference, pd.Series(["alpha beta gamma delta epsilon"]))
    assert share == pytest.approx(0.8)
    assert documents == 1.0

    _, below = token_coverage(reference, pd.Series(["alpha beta gamma epsilon"]))
    assert below == 0.0


def test_token_coverage_without_reference_or_tokens() -> None:
    """Пустой словарь и текст без слов дают нули, а не деление на ноль."""
    assert token_coverage(set(), pd.Series(["bgp сессия"])) == (0.0, 0.0)
    assert token_coverage({"bgp"}, pd.Series([], dtype=object)) == (0.0, 0.0)
    assert token_coverage({"bgp"}, pd.Series(["123 456", ""])) == (0.0, 0.0)


# --- пересечение групп между выборками -------------------------------------


def test_split_crossings_pass_on_grouped_split(assigned: pd.DataFrame) -> None:
    report = _report(checks_split_crossings(assigned))

    assert report["status"] == "ok"
    assert report["value"] == "0 нарушений"


def test_split_crossings_fail_after_moving_one_record(assigned: pd.DataFrame) -> None:
    frame = assigned.copy()
    sizes = frame.groupby("event_id").size()
    shared_event = sizes[sizes > 1].index[0]
    index = frame.index[frame["event_id"] == shared_event][0]
    frame.loc[index, "split"] = _other_split(frame.loc[index, "split"])

    report = _report(checks_split_crossings(frame))

    assert report["status"] == "fail"
    # Перенос одной карточки ломает сразу сценарий и событие.
    assert "scenario_id: 1 групп" in report["value"]
    assert "event_id: 1 групп" in report["value"]


def test_split_crossings_fail_on_shared_text_hash(assigned: pd.DataFrame) -> None:
    """Третий групповой ключ проверяется отдельно: сценарии и события остаются раздельными."""
    frame = assigned.copy()
    train = frame.index[frame["split"] == "train"][0]
    test = frame.index[frame["split"] == "test"][0]
    frame.loc[[train, test], "text_hash"] = "f" * 64

    report = _report(checks_split_crossings(frame))

    assert report["status"] == "fail"
    assert "text_hash: 1 групп" in report["value"]
    assert "scenario_id" not in report["value"]


# --- согласованность события ----------------------------------------------


def test_events_pass_on_generated_set(assigned: pd.DataFrame) -> None:
    report = _report(checks_events(assigned))

    assert report["status"] == "ok"
    assert "расхождений severity с последствием: 0" in report["value"]
    assert "разброс времени больше 30 мин: 0" in report["value"]


def test_events_fail_when_severity_differs_inside_event(assigned: pd.DataFrame) -> None:
    sizes = assigned.groupby("event_id").size()
    event_id = sizes[sizes > 1].index[0]
    rows = assigned.index[assigned["event_id"] == event_id]
    present = set(assigned.loc[rows, "severity"])
    frame = assigned.copy()
    frame.loc[rows[0], "severity"] = next(label for label in SEVERITIES if label not in present)

    report = _report(checks_events(frame))

    assert report["status"] == "fail"
    assert "расхождений severity с последствием: 1" in report["value"]


def test_events_report_mixed_impact_scale_without_failing(assigned: pd.DataFrame) -> None:
    """Разный масштаб влияния в одном событии — справка, а не дефект.

    Карточки каскада относятся к разным сценариям, и каждая сама по себе
    согласована со своим последствием, поэтому проверка обязана остаться ok.
    """
    one_per_scenario = assigned.drop_duplicates("scenario_id")
    first = one_per_scenario.iloc[[0]]
    other_scale = one_per_scenario[
        one_per_scenario["severity"].ne(first["severity"].iloc[0])
        & one_per_scenario["scenario_id"].ne(first["scenario_id"].iloc[0])
    ]
    frame = pd.concat([first, other_scale.iloc[[0]]], ignore_index=True)
    frame["event_id"] = "E999999"
    frame["created_at"] = frame["created_at"].iloc[0]

    report = _report(checks_events(frame))

    assert report["status"] == "ok"
    assert "расхождений severity с последствием: 0" in report["value"]
    assert "событий с разным масштабом влияния: 1" in report["value"]


def test_events_fail_when_cards_of_event_are_far_apart(assigned: pd.DataFrame) -> None:
    sizes = assigned.groupby("event_id").size()
    event_id = sizes[sizes > 1].index[0]
    rows = assigned.index[assigned["event_id"] == event_id]
    frame = assigned.copy()
    shifted = pd.to_datetime(frame.loc[rows[0], "created_at"], format="%Y-%m-%dT%H:%M") + (
        pd.Timedelta(minutes=45)
    )
    frame.loc[rows[0], "created_at"] = shifted.strftime("%Y-%m-%dT%H:%M")

    report = _report(checks_events(frame))

    assert report["status"] == "fail"
    assert "разброс времени больше 30 мин: 1" in report["value"]


@pytest.mark.parametrize(
    ("second_stamp", "expected"),
    [("2026-08-01T10:30", "ok"), ("2026-08-01T10:31", "fail")],
    ids=["ровно 30 минут", "31 минута"],
)
def test_events_time_limit_is_strict(second_stamp: str, expected: str) -> None:
    """Граница в 30 минут включительна: нарушением считается только превышение."""
    frame = pd.DataFrame(
        {
            "event_id": ["E000001", "E000001"],
            "created_at": ["2026-08-01T10:00", second_stamp],
            "severity": ["P2", "P2"],
            "service": ["transport", "transport"],
        }
    )

    report = _report(checks_events(frame))

    assert report["status"] == expected


def test_events_fail_on_unreadable_date(assigned: pd.DataFrame) -> None:
    frame = assigned.copy()
    frame.loc[frame.index[0], "created_at"] = "вчера вечером"

    report = _report(checks_events(frame))

    assert report["status"] == "fail"
    assert "нечитаемых дат: 1" in report["value"]


# --- точные дубли, метки, обязательные поля, ключи -------------------------


def test_duplicates_pass_on_cleaned_set(cleaned: pd.DataFrame) -> None:
    report = _report(checks_duplicates(cleaned))

    assert report["status"] == "ok"
    assert "0 повторов text_hash" in report["value"]


def test_duplicates_fail_when_text_is_copied(cleaned: pd.DataFrame) -> None:
    copy = cleaned.iloc[[0]].copy()
    copy.loc[copy.index[0], "incident_id"] = "I-DUPLICATE"
    frame = pd.concat([cleaned, copy], ignore_index=True)

    report = _report(checks_duplicates(frame))

    assert report["status"] == "fail"
    assert "1 повторов text_hash" in report["value"]


def test_allowed_values_pass_on_cleaned_set(cleaned: pd.DataFrame) -> None:
    report = _report(checks_allowed_values(cleaned))

    assert report["status"] == "ok"
    assert report["value"] == "нарушений: 0"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("severity", "P5"),
        ("root_cause", "unknown"),
        ("split", "holdout"),
        ("rights_status", "restricted"),
        ("language", "de"),
    ],
)
def test_allowed_values_fail_on_unknown_category(
    cleaned: pd.DataFrame, column: str, value: str
) -> None:
    frame = cleaned.copy()
    frame.loc[frame.index[0], column] = value

    report = _report(checks_allowed_values(frame))

    assert report["status"] == "fail"
    assert f"{column}: ['{value}']" in report["decision"]


def test_required_pass_on_cleaned_set(cleaned: pd.DataFrame) -> None:
    report = _report(checks_required(cleaned))

    assert report["status"] == "ok"
    assert "0 пустых значений" in report["value"]


@pytest.mark.parametrize(
    ("column", "value"),
    [("title", ""), ("severity", "   "), ("created_at", ""), ("description", "")],
)
def test_required_fail_on_blank_mandatory_field(
    cleaned: pd.DataFrame, column: str, value: str
) -> None:
    frame = cleaned.copy()
    frame.loc[frame.index[0], column] = value

    report = _report(checks_required(frame))

    assert report["status"] == "fail"
    assert column in report["decision"]


def test_keys_pass_on_cleaned_set(cleaned: pd.DataFrame) -> None:
    report = _report(checks_keys(cleaned))

    assert report["status"] == "ok"
    assert "повторов incident_id: 0, affected_ci: 0, пустых affected_ci: 0" in report["value"]
    assert "расхождений text_hash: 0" in report["value"]


def test_keys_fail_on_blank_ci(cleaned: pd.DataFrame) -> None:
    frame = cleaned.copy()
    frame.loc[frame.index[0], "affected_ci"] = "  "

    report = _report(checks_keys(frame))

    assert report["status"] == "fail"
    assert "пустых affected_ci: 1" in report["value"]


@pytest.mark.parametrize(
    ("column", "message"),
    [("incident_id", "повторов incident_id: 1"), ("affected_ci", "affected_ci: 1")],
)
def test_keys_fail_on_duplicate_key(cleaned: pd.DataFrame, column: str, message: str) -> None:
    frame = cleaned.copy()
    frame.loc[frame.index[1], column] = frame.loc[frame.index[0], column]

    report = _report(checks_keys(frame))

    assert report["status"] == "fail"
    assert message in report["value"]


def test_keys_fail_when_text_hash_does_not_match_description(cleaned: pd.DataFrame) -> None:
    frame = cleaned.copy()
    frame.loc[frame.index[0], "text_hash"] = "0" * 64

    report = _report(checks_keys(frame))

    assert report["status"] == "fail"
    assert "расхождений text_hash: 1" in report["value"]


# --- журнал исключений -----------------------------------------------------


def test_cleaning_log_passes_on_pipeline_output(
    cleaned: pd.DataFrame, cleaning_log: pd.DataFrame
) -> None:
    report = _report(checks_cleaning_log(cleaned, cleaning_log))

    assert report["status"] == "ok"
    assert "1. rights_unknown: 3" in report["value"]
    assert "3. duplicate_text: 3" in report["value"]


def test_cleaning_log_fails_when_record_without_rights_keeps_text(
    cleaned: pd.DataFrame, cleaning_log: pd.DataFrame
) -> None:
    log = cleaning_log.copy()
    log.loc[log["stage"] == 1, "description"] = "полный текст записи без подтверждённых прав"

    report = _report(checks_cleaning_log(cleaned, log))

    assert report["status"] == "fail"
    assert "записей без прав сохранены с текстом" in report["decision"]


def test_cleaning_log_fails_when_excluded_record_stays_in_clean(
    cleaned: pd.DataFrame, cleaning_log: pd.DataFrame
) -> None:
    frame = cleaned.copy()
    frame.loc[frame.index[0], "incident_id"] = cleaning_log["incident_id"].iloc[0]

    report = _report(checks_cleaning_log(frame, cleaning_log))

    assert report["status"] == "fail"
    assert "исключённых записей остались в clean" in report["decision"]


def test_cleaning_log_fails_when_rows_are_missing(
    cleaned: pd.DataFrame, cleaning_log: pd.DataFrame
) -> None:
    short_log = cleaning_log.iloc[:-1]

    report = _report(checks_cleaning_log(cleaned, short_log))

    assert report["status"] == "fail"
    assert f"в журнале {len(short_log)} строк" in report["decision"]


# --- повтор шаблонов текста ------------------------------------------------


def test_templates_do_not_cross_splits_on_grouped_set(cleaned: pd.DataFrame) -> None:
    report = _report(checks_templates(cleaned, template_groups(cleaned)))

    assert report["status"] != "fail"
    assert "в разных выборках: 0" in report["value"]


def test_templates_fail_when_one_card_moves_to_another_split(cleaned: pd.DataFrame) -> None:
    keys = cleaned["description"].map(template_key)
    sizes = keys.value_counts()
    shared = sizes[sizes > 1].index[0]
    index = cleaned.index[keys == shared][0]
    frame = cleaned.copy()
    frame.loc[index, "split"] = _other_split(frame.loc[index, "split"])

    report = _report(checks_templates(frame, template_groups(frame)))

    assert report["status"] == "fail"
    assert "в разных выборках: 1" in report["value"]


# --- proxy-признаки в служебных полях --------------------------------------


def _proxies(plain: float, ci_decile: float) -> pd.DataFrame:
    """Таблица proxy-признаков в формате proxy_table: шесть полей и три метки."""
    rows = [
        {"field": field, "severity": plain, "service": plain, "root_cause": plain}
        for field in ("source_system", "region", "language", "host_region")
    ]
    rows.append(
        {"field": "ci_decile", "severity": ci_decile, "service": ci_decile, "root_cause": ci_decile}
    )
    rows.append({"field": "scenario_id", "severity": 1.0, "service": 1.0, "root_cause": 1.0})
    return pd.DataFrame(rows)


def test_proxy_table_scores_every_audit_field(cleaned: pd.DataFrame) -> None:
    table = proxy_table(cleaned)

    assert table["field"].tolist() == list(AUDIT_FIELDS)
    scores = table[["severity", "service", "root_cause"]].to_numpy()
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    scenario = table.set_index("field").loc["scenario_id"]
    # Сценарий однозначно задаёт сервис и причину: это и есть верхняя граница утечки.
    assert scenario["service"] == pytest.approx(1.0)
    assert scenario["root_cause"] == pytest.approx(1.0)


def test_proxy_table_without_numeric_ci_gives_zero_signal(cleaned: pd.DataFrame) -> None:
    frame = cleaned.copy()
    frame["affected_ci"] = "без номера"

    table = proxy_table(frame).set_index("field")

    # Дециль не строится: остаётся одна категория, связи нет.
    assert table.loc["ci_decile"].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("ci_decile", "expected"),
    [(PROXY_WARN_LEVEL - 0.001, "ok"), (PROXY_WARN_LEVEL, "warn")],
    ids=["ниже порога", "ровно порог"],
)
def test_proxy_threshold_is_strict(ci_decile: float, expected: str) -> None:
    """Порог PROXY_WARN_LEVEL не включается: ровно на нём уже предупреждение."""
    report = _report(checks_proxy(_proxies(0.03, ci_decile)))

    assert report["status"] == expected


def test_proxy_check_reports_ci_decile_leak() -> None:
    report = _report(checks_proxy(_proxies(0.05, 0.62)))

    assert report["status"] == "warn"
    assert "дециль affected_ci: 0.62 (service)" in _plain(report["value"])


def test_proxy_check_ignores_scenario_id_in_the_verdict() -> None:
    """scenario_id заведомо сильный, но в статус не входит: он и так не признак."""
    report = _report(checks_proxy(_proxies(0.03, 0.05)))

    assert report["status"] == "ok"
    assert "scenario_id: 1.00" in _plain(report["value"])


def test_ru_formats_numbers_with_a_decimal_comma() -> None:
    """Таблица проверок идёт в документ, поэтому числа печатаются по-русски."""
    assert ru(0.6171, 2) == "0,62"
    assert ru(0.5) == "0,5"
    assert ru(1.0, 2) == "1,00"
    assert ru(0.0) == "0,0"


# --- загрузка и схема ------------------------------------------------------


def _write_pipeline(tmp_path: Path, raw: pd.DataFrame, cleaned: pd.DataFrame, log: pd.DataFrame,
                    assigned: pd.DataFrame) -> None:
    """Раскладывает шесть файлов пайплайна так, как их ждёт load_frames."""
    parts = {"raw": raw, "clean": cleaned, "log": log}
    for split in SPLIT_ORDER:
        parts[split] = assigned[assigned["split"] == split]
    for name, part in parts.items():
        path = tmp_path / DATA_PATHS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, index=False, encoding="utf-8")


def test_load_frames_reads_pipeline_output(
    tmp_path: Path,
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    assigned: pd.DataFrame,
) -> None:
    _write_pipeline(tmp_path, raw, cleaned, cleaning_log, assigned)

    frames = load_frames(tmp_path)

    assert list(frames) == list(DATA_PATHS)
    # stage приводится к числу: checks_cleaning_log сравнивает его с целыми этапами.
    assert frames["log"]["stage"].dtype.kind == "i"
    assert len(frames["clean"]) == len(cleaned)
    restored = pd.concat([frames[split] for split in SPLIT_ORDER], ignore_index=True)
    assert set(restored["incident_id"]) == set(cleaned["incident_id"])


def test_load_frames_names_missing_files(tmp_path: Path) -> None:
    """Сообщение должно подсказывать, что данные не собраны, а не падать трейсбеком."""
    with pytest.raises(FileNotFoundError, match="Пройдите пайплайн"):
        load_frames(tmp_path)


def test_schema_fail_when_column_is_missing(
    raw: pd.DataFrame, cleaned: pd.DataFrame
) -> None:
    broken = cleaned.drop(columns=["root_cause"])

    report = _report(checks_schema(raw, broken))

    assert report["status"] == "fail"
    assert "root_cause" in report["decision"]


# --- сводный прогон --------------------------------------------------------


def test_run_checks_returns_full_report(medium_frames: dict[str, pd.DataFrame]) -> None:
    """На наборе меньше 6 000 проверки не должны падать: статусы зависят от объёма."""
    results, groups = run_checks(medium_frames)

    assert len(results) == 19
    assert all(set(row) == {"check", "value", "decision", "status"} for row in results)
    assert {row["status"] for row in results} <= {"ok", "warn", "fail"}
    assert len({row["check"] for row in results}) == len(results)
    assert any(row["check"] == "Объём raw и clean" for row in results)
    assert any(row["check"] == "Состав выборок и clean" for row in results)
    assert list(groups.columns) == ["records", "scenarios", "splits"]
    assert groups.index.name == "template"

# --- объём и состав -------------------------------------------------------


def test_volume_fails_when_records_lost(medium_frames: dict[str, pd.DataFrame]) -> None:
    frames = dict(medium_frames)
    frames['clean'] = frames['clean'].iloc[:-1]

    report = _report(checks_volume(frames))

    assert report['status'] == 'fail'
    assert 'нужна пересборка' in report['decision']


def test_split_files_pass_on_consistent_frames(medium_frames: dict[str, pd.DataFrame]) -> None:
    report = _report(checks_split_files(medium_frames))

    assert report['status'] == 'ok'


def test_split_files_fail_on_stale_split(medium_frames: dict[str, pd.DataFrame]) -> None:
    frames = dict(medium_frames)
    frames['test'] = frames['test'].iloc[:-1]

    report = _report(checks_split_files(frames))

    assert report['status'] == 'fail'
    assert 'пересобрать' in report['decision']


# --- каскады, покрытие и поддержка классов --------------------------------


def test_cascades_fail_when_pair_splits(medium_frames: dict[str, pd.DataFrame]) -> None:
    work = pd.concat([medium_frames[split] for split in SPLIT_ORDER], ignore_index=True)
    left, right = CASCADES[0]
    current = work.loc[work['scenario_id'] == right, 'split'].iloc[0]
    work.loc[work['scenario_id'] == right, 'split'] = _other_split(current)

    report = _report(checks_cascades(work))

    assert report['status'] == 'fail'
    assert left in report['value'] and right in report['value']


def test_coverage_reports_missing_service(medium_frames: dict[str, pd.DataFrame]) -> None:
    work = pd.concat([medium_frames[split] for split in SPLIT_ORDER], ignore_index=True)
    work = work[~((work['split'] == 'test') & (work['service'] == 'ims'))]

    report = _report(checks_coverage(work))

    assert report['status'] == 'fail'
    assert 'ims' in report['decision']


def test_class_support_sees_absent_class(medium_frames: dict[str, pd.DataFrame]) -> None:
    work = pd.concat([medium_frames[split] for split in SPLIT_ORDER], ignore_index=True)
    work.loc[work['split'] == 'validation', 'severity'] = 'P2'

    report = _report(checks_class_support(work))

    assert report['status'] == 'warn'
    assert 'нет' in report['value']
    assert 'severity' in report['value']


# --- пустой вход и CLI ----------------------------------------------------


@pytest.mark.parametrize(
    'func',
    [
        checks_required,
        checks_keys,
        checks_duplicates,
        checks_allowed_values,
        checks_time,
        checks_text_length,
        checks_title,
    ],
)
def test_checks_fail_on_empty_frame(func) -> None:
    report = _report(func(pd.DataFrame()))

    assert report['status'] == 'fail'
    assert 'пуст' in report['value']


def test_volume_fails_on_empty_frames() -> None:
    empty = pd.DataFrame()
    report = _report(checks_volume({'raw': empty, 'clean': empty, 'log': empty}))

    assert report['status'] == 'fail'


def test_main_returns_one_without_data(tmp_path: Path) -> None:
    assert main(['--root', str(tmp_path)]) == 1


def test_display_path_keeps_path_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / 'figures' / 'plot.png'
    root = tmp_path / 'repo'

    assert display_path(outside, root) == str(outside)

