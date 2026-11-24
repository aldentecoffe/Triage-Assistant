# Журнал эксперимента

## Идентификация запуска

| Поле | Значение |
|---|---|
| Код эксперимента | E01 |
| Версия модуля | 1.0 |
| Дата прогона | 2026-11-24 |
| Автор прогона | Плотников М. А., учебный проект; прогон выполнен модулем `src/experiment.py` |
| Изменение | сигнал сортировки: пул top-10 переранжируется по вероятности причины карточки |
| Команда | `python src/experiment.py run` |
| Код изменения | `src/experiment.py`, функции `build_reranker`, `rerank_indices`, `evaluate_frame` |
| Отбор | validation, 1 200 запросов |
| Контроль | test после фиксации решения, отдельная команда |

## Конфигурация запуска

```json
{
  "experiment_id": "E01",
  "version": "1.0",
  "baseline": {
    "name": "tfidf_cosine_top3",
    "pool": null,
    "reranker": null
  },
  "candidate": {
    "name": "tfidf_cosine_pool10_cause_rerank",
    "pool": 10,
    "reranker": {
      "model": "LogisticRegression",
      "solver": "lbfgs",
      "C": 1.0,
      "max_iter": 2000,
      "class_weight": "balanced"
    }
  },
  "k": 3,
  "main_metric": "ndcg@3 по шкале v2.0",
  "grading_rule": "2 — та же причина; 1 — та же услуга при другой причине; 0 — иначе",
  "selection_split": "validation",
  "final_split": "test",
  "data_version": "1.1",
  "vectorizer": {
    "analyzer": "word",
    "lowercase": true,
    "ngram_range": [
      1,
      2
    ],
    "min_df": 2,
    "max_features": 20000,
    "sublinear_tf": true
  },
  "seed": null,
  "alpha_rule": "baseline NDCG@3 + 0,10",
  "spread_limit": 0.5
}
```

## Контроль условий

| Условие | Значение |
|---|---|
| Версия данных | 1.1 |
| Словарь TF-IDF | `44c848662e62332db0c179052fa1b8846e3fccb10e0da4c78dcfc601f228574e` |
| Отпечаток конфигурации | `1bc65b563b8aab289759f078e58fef7afec2cc66f3ffcde1239e33052f24152e` |
| Train, SHA-256 | `d641ceb644b0a25be5a869bb1e8e97bc9b786c28b88cb0117a880a4666b071f2` |
| Validation, SHA-256 | `73f5e83f4027c6ee44702e8a9a7541fd9c9404c9641a7991504b38fb87279079` |
| Python | 3.12.10 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.1 |
| numpy | 2.5.5 |
| Платформа | Windows-10-10.0.19045-SP0 |

Случайность не используется: seed не задан, повторный запуск даёт те же оценки, поэтому десять повторов не добавили бы независимых наблюдений.

## Результаты

| Файл | Что внутри |
|---|---|
| `docs/lab9/results/summary.json` | сводка, условия решения, отпечатки, окружение |
| `docs/lab9/results/per_query.csv` | метрики каждого запроса до и после |
| `docs/lab9/results/segments.csv` | разрезы из сравнительной таблицы |
| `docs/lab9/results/config.json` | конфигурация запуска |
| `docs/lab9/results/decision.json` | зафиксированное решение и проверки |
| `docs/lab9/results/test_summary.json` | контроль на test после решения |
| `docs/lab9/results/variants.json` | сравнение четырёх сигналов на validation |

Сверка baseline с работой 8: NDCG@3 0,413 против 0,413, разница 0,000, совпадает.

## Решение

Решение E01: **не включать**. Причина: прирост 0,067 меньше заданных 0,10, и попадание причины в top-3 у P1 обнулилось. Проверки: gain — не выполнено, p1_cause — не выполнено, p1_ndcg — выполнено, empty — выполнено, spread — выполнено. Расшифровка условий — в `decision.md`.

Контроль на test: baseline 0,325, кандидат 0,317, прирост -0,008. Test не влиял на выбор.
