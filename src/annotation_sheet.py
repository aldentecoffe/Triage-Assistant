"""Слепые листы пилотной разметки для ЛР 6.

Собирает выборку карточек из validation, перемешивает и раскладывает в листы,
где разметчику видны только номер карточки и описание. Метки, сервис, причина,
`scenario_id` и `event_id` в лист не попадают: иначе проверялась бы не
инструкция, а память о сгенерированных метках.

Два листа: первый проход по `--count` карточкам, второй (отложенный) — по
`--repeat` карточкам из первого листа, в другом порядке и с другими номерами.
Ключ «номер карточки → incident_id» лежит отдельно и открывается только после
второго прохода. Рядом пишутся `form_a.html` и `form_b.html`: локальная форма
с выпадающими списками, сохранением прогресса в браузере и выгрузкой CSV в том
же формате — заполнять быстрее, чем таблицу, и труднее испортить кодировку.

Запуск из корня репозитория:
    python src/annotation_sheet.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from string import Template

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


FORM_TEMPLATE = Template(
    """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>$title</title>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; font: 15px/1.5 "Segoe UI", system-ui, sans-serif; background: #f6f6f4; color: #1c1c1a; }
  header { display: flex; align-items: center; gap: 12px; padding: 12px 20px; background: #1c1c1a; color: #f6f6f4; position: sticky; top: 0; }
  header .title { font-weight: 600; }
  header .grow { flex: 1; }
  main { display: grid; grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr); gap: 20px; padding: 20px; max-width: 1180px; margin: 0 auto; }
  .card { background: #fff; border: 1px solid #d9d9d4; border-radius: 10px; padding: 20px; }
  .meta { font: 13px ui-monospace, monospace; color: #6b6b63; margin-bottom: 8px; }
  .text { font-size: 17px; margin: 0 0 18px; }
  .fields { display: grid; gap: 12px; }
  label { display: grid; gap: 4px; font-size: 13px; color: #4a4a44; }
  select, input, textarea { font: inherit; padding: 7px 8px; border: 1px solid #c9c9c2; border-radius: 7px; background: #fff; color: inherit; }
  textarea { min-height: 54px; resize: vertical; }
  .nav { display: flex; gap: 10px; margin-top: 18px; }
  button { font: inherit; padding: 9px 16px; border-radius: 8px; border: 1px solid #1c1c1a; background: #1c1c1a; color: #f6f6f4; cursor: pointer; }
  button.ghost { background: transparent; color: inherit; }
  button:disabled { opacity: .4; cursor: default; }
  aside { font-size: 13px; }
  aside h2 { font-size: 14px; margin: 0 0 6px; }
  aside section { background: #fff; border: 1px solid #d9d9d4; border-radius: 10px; padding: 14px; margin-bottom: 12px; }
  aside ul { margin: 6px 0 0; padding-left: 18px; }
  aside li { margin-bottom: 4px; }
  code { font: 12px ui-monospace, monospace; background: #efefeb; padding: 1px 4px; border-radius: 4px; }
  .empty { border-color: #b4552d; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <span class="title">$title</span>
  <span id="progress">0 / $total</span>
  <span class="grow"></span>
  <button id="export">Скачать CSV</button>
</header>
<main>
  <section class="card" id="card-box">
    <div class="meta">Карточка <span id="card-id"></span></div>
    <p class="text" id="card-text"></p>
    <div class="fields">
      <label>Критичность
        <select id="severity">
          <option value="">— выберите —</option>
          <option value="P1">P1 — ключевая функция не выполняется, обходного пути нет</option>
          <option value="P2">P2 — не выполняется для части или отдельная функция/интеграция</option>
          <option value="P3">P3 — задержка, очередь, обходной путь есть</option>
          <option value="P4">P4 — абоненты не затронуты</option>
          <option value="спорный">спорный — правило не даёт ответа</option>
        </select>
      </label>
      <label>Сервис
        <select id="service">
          <option value="">— выберите —</option>
          <option value="transport">transport — BGP, префиксы, FIB, кольцо агрегации</option>
          <option value="platform">platform — Kubernetes, поды, деплои, пул адресов</option>
          <option value="database">database — СУБД, реплика, транзакции, схема</option>
          <option value="ims">ims — IMS, VoLTE, регистрация, HSS, Diameter</option>
          <option value="billing">billing — начисления, баланс, платежи, сверка</option>
          <option value="bss">bss — BSS/OSS, CRM, заказы, тарифы, справочник</option>
          <option value="external">external — поставщик, интеграция, адаптер, статусы</option>
        </select>
      </label>
      <label>Категория причины
        <select id="root_cause">
          <option value="">— выберите —</option>
          <option value="change">change — изменили настройки, правила, справочник, схему, сертификат</option>
          <option value="capacity">capacity — исчерпан лимит, пул, очередь, процессор</option>
          <option value="hardware">hardware — отказал узел, нода, карта, диск</option>
          <option value="software">software — компонент падает, сбрасывает, считает неверно</option>
          <option value="network">network — маршрутизация, TTL, потери, BGP</option>
          <option value="external_dep">external_dep — виноват внешний поставщик или его API</option>
        </select>
      </label>
      <label>Уверенность
        <select id="confidence">
          <option value="">—</option>
          <option value="3">3 — правило применимо без оговорок</option>
          <option value="2">2 — правило применимо, но в тексте не хватает детали</option>
          <option value="1">1 — правило ответа не даёт, метка по догадке</option>
        </select>
      </label>
      <label>Комментарий
        <textarea id="comment" placeholder="Обязателен для «спорного» и уверенности 1"></textarea>
      </label>
    </div>
    <div class="nav">
      <button class="ghost" id="prev">Назад</button>
      <button id="next">Дальше</button>
    </div>
  </section>
  <aside>
    <section>
      <h2>Коротко о критичности</h2>
      <ul>
        <li><code>P1</code> — голос и регистрация в сети, платежи или магистраль не работают, обходного пути нет.</li>
        <li><code>P2</code> — функция не выполняется для части, недоступна отдельная функция или интеграция, клиент видит ошибку.</li>
        <li><code>P3</code> — деградация, задержка, очередь, часть ассортимента; обходной путь есть.</li>
        <li><code>P4</code> — абоненты не затронуты: мониторинг, отчётность, справочники, самовосстановление.</li>
        <li><code>спорный</code> — правило молчит; опишите оба чтения в комментарии.</li>
      </ul>
    </section>
    <section>
      <h2>Порядок</h2>
      <ul>
        <li>Читайте только описание карточки.</li>
        <li>Сначала критичность по последствию, не по симптому и не по величине метрики.</li>
        <li>Потом сервис по названным системам, затем причина по тому, что сломалось.</li>
        <li>Уверенность 3 — без оговорок, 2 — не хватает детали, 1 — догадка.</li>
      </ul>
    </section>
    <section>
      <h2>Что дальше</h2>
      <p>Когда все карточки заполнены, нажмите «Скачать CSV» и сохраните файл как <code>$csv_name</code> в папку <code>data/annotation</code>, заменив исходный. Прогресс хранится в браузере, можно закрыть вкладку и вернуться.</p>
    </section>
  </aside>
</main>
<script>
const CARDS = $cards_json;
const STORAGE_KEY = "$storage_key";
const FIELDS = ["severity", "service", "root_cause", "confidence", "comment"];

function emptyAnswers() {
  return CARDS.map(function (card) {
    return { card: card.card, description: card.description, severity: "", service: "", root_cause: "", confidence: "", comment: "" };
  });
}

function loadAnswers() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyAnswers();
    const saved = JSON.parse(raw);
    if (!Array.isArray(saved) || saved.length !== CARDS.length) return emptyAnswers();
    return saved;
  } catch (error) {
    return emptyAnswers();
  }
}

let answers = loadAnswers();
let index = 0;

function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(answers)); } catch (error) {}
}

function filled(answer) {
  return ["severity", "service", "root_cause"].every(function (field) { return answer[field]; });
}

function render() {
  const answer = answers[index];
  document.getElementById("card-id").textContent = answer.card;
  document.getElementById("card-text").textContent = answer.description;
  FIELDS.forEach(function (field) {
    document.getElementById(field).value = answer[field] || "";
  });
  const done = answers.filter(filled).length;
  document.getElementById("progress").textContent = done + " / " + answers.length;
  document.getElementById("prev").disabled = index === 0;
  document.getElementById("next").disabled = index === answers.length - 1;
  document.getElementById("card-box").classList.toggle("empty", !filled(answer));
}

FIELDS.forEach(function (field) {
  document.getElementById(field).addEventListener("input", function (event) {
    answers[index][field] = event.target.value;
    save();
    render();
  });
});

function move(step) {
  index = Math.min(Math.max(index + step, 0), answers.length - 1);
  render();
}

document.getElementById("prev").addEventListener("click", function () { move(-1); });
document.getElementById("next").addEventListener("click", function () { move(1); });

document.addEventListener("keydown", function (event) {
  const tag = document.activeElement.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT") return;
  if (event.key === "ArrowRight") { move(1); }
  if (event.key === "ArrowLeft") { move(-1); }
  if (["1", "2", "3", "4"].includes(event.key)) {
    answers[index].severity = "P" + event.key;
    save();
    render();
  }
});

function csvCell(value) {
  const text = (value === undefined || value === null) ? "" : String(value);
  return /[",\\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}

document.getElementById("export").addEventListener("click", function () {
  const left = answers.filter(function (answer) { return !filled(answer); }).length;
  if (left > 0 && !confirm("Не заполнено карточек: " + left + ". Скачать как есть?")) return;
  const header = ["card", "description", "severity", "service", "root_cause", "confidence", "comment"];
  const lines = [header.join(",")];
  answers.forEach(function (answer) {
    lines.push(header.map(function (field) { return csvCell(answer[field]); }).join(","));
  });
  const blob = new Blob(["\\ufeff" + lines.join("\\r\\n") + "\\r\\n"], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "$csv_name";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

render();
</script>
</body>
</html>
"""
)


def build_form(sheet: pd.DataFrame, title: str, csv_name: str, storage_key: str) -> str:
    """Собирает локальную HTML-форму: описание карточки, метки, прогресс и экспорт CSV."""
    cards = [
        {"card": row["card"], "description": row["description"]}
        for _, row in sheet.iterrows()
    ]
    payload = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    return FORM_TEMPLATE.substitute(
        title=title,
        total=len(cards),
        cards_json=payload,
        csv_name=csv_name,
        storage_key=storage_key,
    )


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

    forms = {
        "form_a.html": build_form(
            sheet_a, f"Проход A · {len(sheet_a)} карточек", "pass_a.csv", "annotation-pass-a"
        ),
        "form_b.html": build_form(
            sheet_b, f"Проход B · {len(sheet_b)} карточек", "pass_b.csv", "annotation-pass-b"
        ),
    }
    for name, markup in forms.items():
        path = Path(args.outdir) / name
        path.write_text(markup, encoding="utf-8")
        print(f"форма: {path}")

    print(f"первый проход: {len(sheet_a)} карточек, сценариев {sample['scenario_id'].nunique()}")
    print(f"повторный проход: {len(sheet_b)} карточек из первого листа")
    print(f"файлы: {paths['pass_a.csv']}, {paths['pass_b.csv']}, ключ {paths['key.csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
