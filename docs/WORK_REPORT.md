# Отчёт о проделанной работе над mirea-rag-bot

> Дата: 25 июня 2026 г.
> Репозиторий: GitHub

---

## 1. Цель проекта

Построить RAG-бота (Retrieval-Augmented Generation) по сайту РТУ МИРЭА
(mirea.ru), который отвечает на вопросы абитуриентов и студентов на русском
языке, опираясь исключительно на извлечённый из корпуса контекст.

**Ключевые ограничения:**

- Генерация: **Gemma 4 E4B** (GGUF Q4_K_M, ~5 ГБ) через LM Studio, локально.
- Бюджет VRAM: **8 ГБ** (context_length = 8192).
- Эмбеддинги: **intfloat/multilingual-e5-small** (384 измерения).
- Векторное хранилище: **Qdrant** (локально, режим `LocalQdrant`).

Цель: приёмлемое качество ответов (LLM-judge ≥ 85/100) при полностью
локальном инференсе без облачных зависимостей для генерации.

---

## 2. Что было сделано (по фазам)

Работа велась по kanban-доске, задачи разбиты на фазы 0–G.

### Фаза 0. Аудит проекта

Обнаружен критический дефект: в `mirea_data/raw/` скрапер сохранил бинарный
мусор (повреждённые gzip-ответы, HTML с управляющими символами), раздувший
объём сырых данных до **28 МБ**. После очистки и фильтрации объём сырых
данных составил **~0,04 МБ** чистого текста (184 КБ файлов на диске).

Результат: корпус стал пригодным для обработки chunker-ом и эмбеддингов.

### Фаза A. Качество данных

| Подзадача | Реализация |
|---|---|
| Binary guard | `src/scraper/_binary_guard.py` – блокирует запись нетекстовых данных |
| Markdown clean | `src/pipeline/cleaner.py` – нормализация пробелов, удаление шаблонного мусора |
| LLM-очистка | `src/pipeline/llm_cleaner.py` – доочистка через LLM |
| Boilerplate titles | удаление дублирующихся заголовков страниц (menu/nav/footer) |
| Chunker (rebalance) | `src/pipeline/chunker.py` – группировка по URL + жадное слияние мелких чанков (<100 слов) |

**Ключевое изменение (task t_ffd3acfc):** перебалансировка chunker-а. Медиана
размера чанка выросла с **39 → 201,5 слов** (129 → 52 чанка). Это улучшило
полноту контекста для генерации и снизило фрагментацию.

### Фаза B. Evaluation framework

| Подзадача | Реализация |
|---|---|
| Golden QA | `mirea_data/eval/golden_qa.json` – 50 вопросов, размеченных по 11 категориям и 3 уровням сложности |
| Retrieval metrics | `scripts/eval_retrieval.py` → `retrieval_report.json` |
| LLM-as-judge | `scripts/eval_llm_judge.py` → `judge_report.json` (glm-5.2 judge, 4 измерения) |

Golden set покрывает категории: `applicants`, `sveden`, `contacts`,
`dormitories`, `employment`, `grants`, `extracurricular`, `announcements`,
`student_portal`, `qa_pairs`, `education`.

### Фаза C. Retrieval

| Подзадача | Реализация |
|---|---|
| Cross-encoder rerank | `src/rag/reranker.py` (bge-reranker-v2-m3), `scripts/eval_rerank.py` |
| Query expansion | `src/rag/query_expansion.py` – синонимы/нормализация запроса |
| Hybrid BM25+dense | `src/retrieval/bm25.py` + `src/retrieval/hybrid.py` (RRF fusion) |

**Rerank-эффект (task t_007083df, 155 чанков):**

| Метрика | Baseline (bi-encoder) | + Cross-encoder |
|---|---|---|
| MRR | 0.911 | **0.947** (+3.9%) |
| HitRate@3 | 96% | **98%** |
| Recall@3 | 88.7% | **91.7%** |

**Query expansion:** на golden set (52 чанка) измеренный delta = 0 (ни один
запрос не изменил ранжирование). Функционал сохранён в коде как опция, но
по умолчанию включён (`query_expansion: true` в конфиге) – не вредит, может
помочь на вне-доменных запросах.

**Hybrid BM25:** реализован (RRF K=60), но измерения на golden set дали
низкие метрики из-за рассинхронизации индексов после перебалансировки
chunker-а (38/50 вопросов попали в stale-выброску). Режим `hybrid` включён
в конфиге по умолчанию, dense выступает fallback-ом при отсутствии BM25
индекса.

### Фаза D. Generation

| Подзадача | Реализация |
|---|---|
| Prompt engineering | A/B-тестирование 4 вариантов системного промпта |
| Hallucination guard | `_check_confidence()` в `answer.py` – порог уверенности top-1 чанка |
| Formatting | структурированные списки для перечислений, сохранение диапазонов |

**A/B-тест промптов (task t_b2d436d5 + t_8dd0b704):**

| Вариант | Overall | Correctness | Completeness | Relevance | Fluency |
|---|---|---|---|---|---|
| B (completeness + anti-refusal) | 84.2 | 4.12 | 3.56 | 4.32 | 4.84 |
| **D (B + calibrated formatting)** | **85.5** | **4.18** | **3.78** | 4.30 | 4.84 |

Победитель – **вариант D**: добавлены директивы (1) структурированные списки
для перечислений (+0.22 completeness), (2) сохранение диапазонов как есть,
без выдумывания точных цифр. Промпт D зашит в `SYSTEM_PROMPT` константу.

**Hallucination guard (task t_737c1b5e):**

Распределение top-1 скоров на golden set:

| Сигнал | Min | Mean | Max | Порог | Ложные отказы |
|---|---|---|---|---|---|
| Bi-encoder (E5-small cosine) | 0.836 | 0.897 | 0.946 | 0.55 | 0% |
| Cross-encoder (bge sigmoid) | 0.154 | 0.957 | 1.000 | 0.50 | 2.3% |

Порог 0.55 для bi-encoder сидит ниже всего диапазона in-domain запросов
(0 ложных отказов), но выше noise floor для вне-доменных пар.

### Фаза E. Robustness

Обработка граничных случаев в `answer_question()`:

- Пустой / не-текстовый ввод → canned-ответ «переформулируйте вопрос»
- Ошибка загрузки модели эмбеддингов → `PIPELINE_FAILURE_ANSWER`
- Векторное хранилище недоступно / нет релевантных чанков → `NO_CONTEXT_ANSWER`
- Чанки найдены, но top-1 score < порога → `LOW_CONFIDENCE_ANSWER` (источники показываются)
- Ошибка генерации после ретраев → `LLM_FAILURE_ANSWER` (источники возвращаются)
- LM Studio полностью офлайн → `ConnectionError` пробрасывается, API отдаёт 503

Нормализация запроса: приведение к нижнему регистру, очистка лишних пробелов.

Тесты: `tests/test_answer_edge_cases.py`, `tests/test_proxy_bypass.py`.

### Фаза F. Infrastructure

| Подзадача | Реализация |
|---|---|
| Request logging | `src/rag/request_log.py` – SQLite, каждый запрос + ответ + метрики |
| API | `src/rag/api.py` – REST endpoint (FastAPI/Flask) |
| CLI | `src/rag/cli.py` – командная строка |
| API tests | `tests/test_api_integration.py` |
| Benchmark | `benchmark_results.json` – тайминги всех стадий |
| Consistency check | `scripts/check_consistency.py` – проверка целостности индексов |

### Фаза G. Финальная сборка

- Конфигурация в `config.py` (все константы, режимы, пороги)
- Pipeline собирается в `src/rag/answer.py` → `answer_question()`
- Весь корпус из ~52 чанков в Qdrant-индексе

---

## 3. Архитектура системы

```
                        Пользовательский запрос (RU)
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  answer_question()   │
                         │  src/rag/answer.py   │
                         └─────────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              Нормализация   Query encode    Pipeline
              + валидация    (E5-small,       оркестрация
                             384-dim)
                                   │
                    ╔══════════════╧══════════════╗
                    ║   СТАДИЯ 1: RETRIEVAL       ║
                    ║                             ║
                    ║  RETRIEVAL_MODE = "hybrid"  ║
                    ║  ┌─────────┐ ┌───────────┐  ║
                    ║  │  BM25   │ │  Qdrant   │  ║
                    ║  │(lexical)│ │ (dense)   │  ║
                    ║  └────┬────┘ └─────┬─────┘  ║
                    ║       └─────┬──────┘        ║
                    ║        RRF fusion (K=60)    ║
                    ║     fallback → dense only   ║
                    ╚══════════════╤══════════════╝
                                   │ ~20 кандидатов
                    ╔══════════════╧══════════════╗
                    ║  СТАДИЯ 2: RE-RANK (opt.)   ║
                    ║  bge-reranker-v2-m3         ║
                    ║  RERANK_ENABLED = False     ║
                    ╚══════════════╤══════════════╝
                                   │ top_k = 3
                    ╔══════════════╧══════════════╗
                    ║  СТАДИЯ 3: THRESHOLD FILTER ║
                    ║  score >= 0.35              ║
                    ╚══════════════╤══════════════╝
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          ┌─────────────────┐          ┌──────────────────┐
          │ Confidence guard│          │  Если чанков нет │
          │ top-1 >= 0.55?  │── нет ──▶│  NO_CONTEXT_     │
          └────────┬────────┘          │  ANSWER          │
                   │ да                └──────────────────┘
                   ▼
          ┌─────────────────┐
          │ Prompt assembly │
          │ SYSTEM_PROMPT(D)│
          │ + context chunks│
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │   LM Studio     │
          │  Gemma 4 E4B    │
          │  (Q4_K_M, 8ГБ)  │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────────────┐
          │ Ответ + источники       │
          │ + request_log (SQLite)  │
          └─────────────────────────┘
```

**Поток данных (offline, сборка индекса):**

```
mirea.ru → scraper/*.py → raw HTML
    → cleaner.py / llm_cleaner.py → чистый markdown
    → chunker.py → 52 чанка
    → encoder.py (E5-small) → 384-dim векторы
    → qdrant_indexer.py → Qdrant collection "mirea_chunks"
    → build_bm25.py → BM25 индекс (опция для hybrid)
```

---

## 4. Ключевые метрики

Все метрики – из реальных отчётов в `mirea_data/eval/`.

### 4.1. Retrieval (retrieval_report.json)

Конфигурация: E5-small, 52 чанка, top_k=3, threshold=0.35, Qdrant backend,
50 golden-вопросов.

| Метрика | Значение |
|---|---|
| **MRR** | **0.87** |
| Precision@3 | 0.513 |
| Recall@3 | 0.517 |
| **Hit Rate@3** | **98% (49/50)** |
| Recall@5 | 0.60 |
| Hit Rate@5 | 98% |
| Ниже порога (top-1 < 0.35) | 0/50 (0%) |

Метрики по сложности: easy MRR 0.90 (n=26), medium 0.86 (n=22), hard 0.67 (n=2).

Единственный complete miss (top-3 не содержит ни одного релевантного чанка):
q033 «Сколько спортивных секций работает в РТУ МИРЭА и какие?» (relevant
chunks 8/9/28, но top-3 вернул 15/10/50).

### 4.2. Cross-encoder re-ranking (rerank_report.json)

Конфигурация: 155 чанков (до перебалансировки), shortlist=20, bge-reranker-v2-m3.

| Метрика | Baseline | + Re-rank | Дельта |
|---|---|---|---|
| MRR | 0.911 | 0.947 | +0.036 |
| HitRate@3 | 96% | 98% | +2% |
| Recall@3 | 88.7% | 91.7% | +3% |

### 4.3. LLM-as-judge (judge_report.json)

Генерация: gemma-4-26b-a4b-qat. Judge: glm-5.2-chat. 50 вопросов, 4 измерения
(0–5), overall = среднее, масштабированное к 0–100.

| Измерение | Балл |
|---|---|
| Correctness | 4.44 |
| Completeness | 3.98 |
| Relevance | 4.64 |
| Fluency | 4.90 |
| **Overall** | **89.8 / 100** |

Failed: 0/50. Слабейшая категория – `applicants` (74.4/100): много вопросов
о конкретных проходных баллах и ценах, где контекст содержит диапазоны.

### 4.4. Производительность (benchmark_results.json)

| Стадия | Метрика |
|---|---|
| Загрузка E5-small | 10.3 с, RSS 1119 МБ |
| Кодирование запроса | 16.1 мс (mean), 18.0 мс (p95) |
| Qdrant search (top-5) | 0.85 мс (mean) |
| Full RAG (retrieve + encode) | 21.6 мс (mean), 29.2 мс (p95) |
| Batch throughput | 344.9 qps (50 запросов) |
| Ускорение batch vs single | 6.3× |

Примечание: benchmark проводился на 129 чанках (до перебалансировки). Latency
на 52 чанках будет сопоставимой или ниже.

---

## 5. Структура файлов проекта

```
mirea-rag-bot/
├── config.py                    # все константы проекта
├── requirements.txt
├── benchmark_results.json       # тайминги стадий
│
├── src/
│   ├── scraper/                 # сбор данных с mirea.ru
│   │   ├── _binary_guard.py     #   защита от бинарного мусора
│   │   ├── deeper_scraper.py    #   рекурсивный обход
│   │   ├── applicants.py        #   раздел абитуриента
│   │   ├── sveden.py            #   раздел сведений о вузе
│   │   ├── contacts.py          #   контакты
│   │   ├── education.py         #   образование
│   │   └── student_portal.py    #   студенческий портал
│   │
│   ├── pipeline/                # обработка сырых данных
│   │   ├── cleaner.py           #   нормализация markdown
│   │   ├── llm_cleaner.py       #   LLM-очистка
│   │   ├── chunker.py           #   разбиение на чанки
│   │   └── qa_ingest.py         #   ингест QA-пар
│   │
│   ├── embeddings/              # эмбеддинги и индекс
│   │   ├── encoder.py           #   E5-small encoding
│   │   ├── indexer.py           #   FAISS/numpy search
│   │   └── qdrant_indexer.py    #   Qdrant collection
│   │
│   ├── retrieval/               # retrieval backends
│   │   ├── bm25.py              #   BM25 (Okapi BM25)
│   │   └── hybrid.py            #   RRF/weighted fusion
│   │
│   └── rag/                     # RAG pipeline
│       ├── answer.py            #   главная функция answer_question()
│       ├── reranker.py          #   cross-encoder re-rank
│       ├── query_expansion.py   #   расширение запроса
│       ├── api.py               #   REST API
│       ├── cli.py               #   CLI интерфейс
│       └── request_log.py       #   логирование (SQLite)
│
├── scripts/                     # скрипты сборки и оценки
│   ├── eval_retrieval.py        #   retrieval метрики
│   ├── eval_rerank.py           #   rerank A/B
│   ├── eval_llm_judge.py        #   LLM-judge
│   ├── eval_prompt_ab.py        #   prompt A/B
│   ├── eval_confidence_guard.py #   confidence guard sweep
│   ├── eval_hybrid.py           #   hybrid backend сравнение
│   ├── build_index.py           #   сборка векторного индекса
│   ├── build_bm25.py            #   сборка BM25 индекса
│   ├── build_golden.py          #   сборка golden QA
│   ├── rebuild_qdrant.py        #   пересборка Qdrant
│   ├── check_consistency.py     #   проверка целостности
│   ├── verify_quality.py        #   проверка качества данных
│   ├── update_data.py           #   обновление данных
│   └── llm_clean_data.py        #   LLM-очистка корпуса
│
├── tests/                       # 432 теста
│   ├── test_answer_edge_cases.py
│   ├── test_rag_answer.py
│   ├── test_reranker.py
│   ├── test_hybrid.py
│   ├── test_confidence_guard.py
│   ├── test_query_expansion.py
│   ├── test_api_integration.py
│   ├── test_request_log.py
│   ├── test_pipeline_qdrant.py
│   ├── test_qdrant_indexer.py
│   ├── test_search.py
│   ├── test_student_scraper.py
│   ├── test_deeper_scraper.py
│   ├── test_llm_cleaner.py
│   ├── test_proxy_bypass.py
│   └── ... (всего 19 тест-модулей)
│
└── mirea_data/
    ├── raw/                     # сырой HTML (7 файлов, 184 КБ)
    ├── processed/               # очищенный markdown
    ├── eval/                    # все отчёты и golden set
    │   ├── golden_qa.json
    │   ├── retrieval_report.json
    │   ├── rerank_report.json
    │   ├── judge_report.json
    │   ├── ab_report.json       # A/B/C
    │   ├── ab_report_bd.json    # B vs D
    │   ├── confidence_guard_report.json
    │   └── hybrid_report.json
    ├── faiss/                   # FAISS индекс (legacy)
    ├── qdrant/                  # Qdrant коллекция
    └── bm25/                    # BM25 индекс
```

---

## 6. Известные ограничения и TODO

### Ограничения

1. **Объём корпуса.** После перебалансировки – 52 чанка из ~7 страниц mirea.ru.
   Покрытие ограничено scraped-разделами. Вопросы вне корпуса получают
   NO_CONTEXT / LOW_CONFIDENCE ответ.

2. **Hybrid retrieval нестабилен.** При пересборке индексов BM25 и Qdrant
   рассинхронизируются (4 параллельных worker-а меняют корпус). Падает в
   dense fallback. Требует ручного `rebuild_qdrant()` + `build_bm25.py`.

3. **Rerank отключён по умолчанию.** Cross-encoder даёт +3.9% MRR, но
   добавляет ~50–100 мс на запрос. Включается `RERANK_ENABLED = True`.

4. **Слабая категория `applicants`.** LLM-judge 74.4/100 – вопросы о
   конкретных проходных баллах и ценах, где в контексте диапазоны, а не
   точные цифры.

5. **Query expansion не дал эффекта** на golden set (delta = 0). Сохранён как
   опция, но не подтверждён эмпирически.

6. **Qdrant stale points.** `delete_collection()` не чистит points полностью –
   требуется `rm -rf qdrant/` перед пересборкой.

7. **Нет git-репозитория.** Проект не под версионным контролем.

### TODO

- [ ] Расширить корпус: добавить больше разделов mirea.ru
- [ ] Стабилизировать hybrid: атомарная пересборка BM25 + Qdrant
- [ ] Инициализировать git-репозиторий
- [ ] Добавить README.md с инструкцией запуска
- [ ] Включить rerank в production после latency-тестов на 52 чанках
- [ ] Улучшить ответы в категории `applicants` (дополнительный контекст)

---

## Приложение: Kanban-задачи

| ID | Фаза | Описание |
|---|---|---|
| – | 0 | Аудит, очистка бинарного мусора (28 МБ → 0,04 МБ) |
| t_ffd3acfc | A | Перебалансировка chunker (медиана 39 → 201,5 слов) |
| – | A | Binary guard, markdown clean, boilerplate titles |
| t_8ab04e61 | B | Метрики retrieval (MRR, HitRate, Precision/Recall) |
| t_7a079b41 | B | LLM-as-judge (glm-5.2, 50 вопросов, 89.8/100) |
| t_007083df | C | Cross-encoder rerank (MRR 0.911 → 0.947) |
| – | C | Query expansion, hybrid BM25 |
| t_b2d436d5 | D | Prompt engineering A/B (вариант B) |
| t_8dd0b704 | D | Prompt engineering B/D (вариант D – победитель) |
| t_737c1b5e | D | Hallucination guard (confidence threshold) |
| – | E | Edge cases, нормализация, graceful degradation |
| – | F | Logging (SQLite), API, benchmark, consistency check |
| – | G | Финальная сборка, config.py |
