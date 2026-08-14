# МИРЭА RAG Bot

RAG-система для ответов на вопросы о РТУ МИРЭА по данным, собранным с mirea.ru и student.mirea.ru.

## Архитектура

Система состоит из трёх плоскостей: подготовка данных, retrieval+generation пайплайн и API-слой.

### Плоскость данных (offline)

```
mirea.ru / student.mirea.ru          VIKA intents dataset (1384 QA-пары)
        │                                        │
        ▼                                        ▼
  [Скраперы 6 шт] ──► raw/*.json          vika_chunks.json
        │                                        │
        ▼                                        │
  [Cleaner] ──► processed/cleaned.json           │
  (HTML→markdown→boilerplate→ws, sha256-dedup)   │
        │                                        │
        ▼                                        │
  [Chunker] ──► processed/chunks.json (75 чанков,│
   токенайзер E5, 300 ток., overlap 50)          │
        │                                        │
        ▼                                        ▼
  [Encoder E5-small 384d] ──► faiss/vectors.npy + metadata.json
        │                          (позиционный doc-id контракт)
        ▼
  [Qdrant embedded] 1376 points ◄── [BM25 index 1459 docs]
   mirea_data/qdrant/                 mirea_data/bm25/bm25_index.pkl
```

`scripts/update_data.py` прогоняет весь пайплайн: бэкап Qdrant в сиблинг-директорию →
пересборка → копирование metadata.json → rebuild BM25 (все три индекса делят
позиционные id, поэтому пересобираются вместе).

### Пайплайн ответа (`src/rag/answer.py`, online)

```
question
  │ 1. Валидация: тип, control-chars, длина ≤500, осмысленность
  │ 2. Нормализация: раскрытие 18 аббревиатур МИРЭА, раскладка QWERTY→ЙЦУКЕН,
  │    пунктуация, whitespace
  │ 3. Query rewriting: короткий вопрос/местоимения склеиваются
  │    с предыдущим вопросом из history
  ▼
[E5 encode] (prefix "query: ")
  ▼
[Hybrid retrieval]  BM25 + Dense, RRF-фузия (k=60), пул 3×top_k
  │   dense-cosine прокидывается через фузию как dense_score —
  │   пороги считаются по абсолютной шкале E5, а не по fused-рангу
  ▼
[Секционный фильтр]  INFO_SECTIONS (10 инфо-секций);
  │   исключение: VIKA-чанки с dense ≥ VIKA_DENSE_GATE (0.88) проходят —
  │   часть фактов живёт только в VIKA QA (общежития, корпуса)
  ▼
[Пороги]  RELEVANCE ≥ 0.35 (по dense_score) → top_k = 5
  │   Hallucination guard: top-1 ≥ 0.55 (bi) / 0.50 (cross-encoder),
  │   иначе отказ с показанными источниками
  ▼
[Генерация]  Gemma 4 E4B (LM Studio :1234), ретраи ×3, анти-инъекции history
  ▼
[Пост-обработка]  code-fence, метки «Ответ:», em-dash→en-dash → answer + sources
```

### API-слой

```
Docker-контейнер :8080
  ├── FastAPI  POST /ask (генерация и лог — в threadpool, event loop свободен)
  │            GET /health, GET /analytics
  │            GET / — веб-чат (static/index.html)
  ├── SQLite requests.db — лог вопросов/ответов/уверенности
  └── Qdrant embedded + BM25 + E5 (внутри контейнера)
        ▲
        │ host.docker.internal:1234
  LM Studio (хост): Gemma 4 E4B IT Q4_K_M
```

## Требования

- **Python 3.11+**
- **LM Studio** с Gemma 4 E4B Q4_K_M (~5 ГБ)
- ~6.7 ГБ RAM/VRAM суммарно

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Скачать и загрузить модель

LM Studio → Models → Search `gemma-4-e4b-it-GGUF` → Download **Q4_K_M** (~5 ГБ)
→ Developer → Start Server (порт 1234)

### 2. Собрать данные

```bash
# Скрапинг (требует доступ к mirea.ru)
python -m src.scraper.main

# Пайплайн очистки + чанкинг + индексация
python scripts/build_index.py

# BM25 индекс (для hybrid retrieval)
python scripts/build_bm25.py
```

### 3. Запустить API

```bash
uvicorn src.rag.api:app --host 0.0.0.0 --port 8080 --reload
```

### 4. Тест CLI

```bash
python -m src.rag.cli
```

## API

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/ask` | POST | `{question: str}` → `{answer, sources, confidence, generation_time}` |
| `/health` | GET | Статус: E5, Qdrant, LM Studio |
| `/analytics` | GET | Статистика за N дней (`?days=7`) |

### Пример запроса

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Где находится главный корпус МИРЭА?"}'
```

## Ключевые модули

| Файл | Назначение |
|------|------------|
| `config.py` | Конфигурация: модели, пороги, пути |
| `src/pipeline/cleaner.py` | Очистка: HTML → markdown → boilerplate → whitespace |
| `src/pipeline/chunker.py` | Чанкинг по предложениям с E5-токенизатором + overlap |
| `src/embeddings/encoder.py` | E5-small эмбеддинги (384d) |
| `src/embeddings/indexer.py` | Qdrant поиск + сборка индекса |
| `src/retrieval/bm25.py` | BM25 лексический поиск |
| `src/retrieval/hybrid.py` | BM25 + Dense RRF fusion |
| `src/rag/answer.py` | Ядро: encode → retrieve → guard → generate |
| `src/rag/query_expansion.py` | Раскрытие 18 аббревиатур МИРЭА |
| `src/rag/query_normalization.py` | Нормализация: whitespace, punctuation, keyboard layout fix |
| `src/rag/reranker.py` | Cross-encoder re-ranking (опционально, VRAM) |
| `src/rag/request_log.py` | SQLite логирование + аналитика |
| `src/rag/api.py` | FastAPI сервер |
| `src/rag/cli.py` | Микро-REPL для тестов |

## Retrieval

Гибридный retrieval по умолчанию: BM25 (лексический, точные коды/числа/имена) +
E5 Dense (семантика) через RRF-фузию, пул кандидатов 3×top_k:

```python
# config.py
RETRIEVAL_MODE = "hybrid"
HYBRID_FUSION = "rrf"
HYBRID_RRF_K = 60
HYBRID_DENSE_WEIGHT = 0.5
```

**Важно про шкалы.** RRF-скоры минмакс-нормированы по пулу кандидатов (top-1
всегда 1.0), поэтому абсолютные пороги по ним не работают. Гибрид прокидывает
dense-cosine каждого кандидата через фузию как `dense_score` — именно по нему
считаются `RELEVANCE_THRESHOLD` и `CONFIDENCE_THRESHOLD`.

**Секционный фильтр.** ~95% корпуса — диалоговые qa_vika-чанки; для фактических
вопросов они шум. После фузии не-информационные секции отбрасываются
(`INFO_SECTIONS`), но VIKA-чанк с `dense_score ≥ VIKA_DENSE_GATE` проходит:
часть фактов есть только в VIKA QA (общежития, корпуса, навигация).

Cross-encoder re-ranking доступен, но выключен по умолчанию (съедает VRAM вместе с Gemma):

```python
RERANK_ENABLED = False  # True для включения
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
```

## Пороги

Все пороги (кроме fused) оцениваются по абсолютной dense-шкале E5
(in-domain ≈ 0.84–0.95, out-of-domain ≈ 0.30–0.60):

| Параметр | Значение | Назначение |
|----------|----------|------------|
| `RELEVANCE_THRESHOLD` | 0.35 | Минимальный dense_score чанка |
| `CONFIDENCE_THRESHOLD` | 0.55 | Hallucination guard (bi-encoder) |
| `CE_CONFIDENCE_THRESHOLD` | 0.50 | Hallucination guard (cross-encoder) |
| `VIKA_DENSE_GATE` | 0.88 | Пропуск сильных VIKA-чанков сквозь секционный фильтр |
| `SECTION_FILTER_ENABLED` | True | Секционный фильтр шума |
| `SECTION_FILTER_POOL` | 5 | Over-fetch множитель для фильтра |
| `MAX_CONTEXT_CHUNKS` | 5 | Лимит чанков в контексте |
| `CHUNK_SIZE` | 300 | Токенов E5 на чанк |
| `CHUNK_OVERLAP` | 50 | Токенов overlap |

## Скрипты

| Скрипт | Назначение |
|--------|------------|
| `scripts/build_index.py` | Чанки → эмбеддинги → Qdrant |
| `scripts/build_bm25.py` | BM25 индекс из metadata.json |
| `scripts/build_golden.py` | Golden QA dataset |
| `scripts/eval_multihop.py` | Multi-hop + hard-set eval (38 вопросов): `--config baseline\|filtered\|decomposed`, `EVAL_QDRANT_PATH` для запуска рядом с живым API |
| `scripts/verify_quality.py` | 31 проверка качества данных |
| `scripts/verify_search.py` | 21 тест-запрос retrieval |
| `scripts/check_consistency.py` | Синхронизация chunks/Qdrant/golden |
| `scripts/update_data.py` | Полный пайплайн (скрап → индекс + бэкап + rebuild BM25) |
| `scripts/test_lm_studio.py` | Проверка подключения к LM Studio |
