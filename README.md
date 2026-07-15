# МИРЭА RAG Bot

RAG-система для ответов на вопросы о РТУ МИРЭА по данным, собранным с mirea.ru и student.mirea.ru.

## Архитектура

```
mirea.ru / student.mirea.ru
        │
        ▼
  [Скраперы] ──► raw/ (7 JSON: education, applicants, sveden, contacts, qa_pairs, deeper, student_portal)
        │               │
        │
        ▼
  [Cleaner] ──► processed/cleaned.json (dedup, strip HTML/markdown/boilerplate)
        │
        ▼
  [Chunker] ──► processed/chunks.json (300 токенов E5, overlap 50, 75 чанков)
        │
        ▼
  [Encoder] ──► faiss/vectors.npy (75×384, E5-small) + Qdrant (75 points)
        │              + bm25/bm25_index.pkl (2702 термина)
        │
        ▼
  [RAG answer.py] ── query → encode → hybrid retrieve (BM25+Dense RRF)
        │                      → hallucination guard (confidence ≥ 0.55)
        │                      → prompt → LM Studio (Gemma 4 E4B)
        │                      → post-process → answer + sources
        │
        ▼
  FastAPI /ask │ CLI python -m src.rag.cli
```

## Требования

- **Python 3.11+**
- **LM Studio** с Gemma 4 E4B Q4_K_M (~5 ГБ)
- ~6.7 ГБ RAM/VRAM суммарно

## Установка

```bash
pip install sentence-transformers qdrant-client fastapi uvicorn \
    numpy requests beautifulsoup4 lxml pymupdf rank_bm25
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
| `src/embeddings/indexer.py` | Qdrant + FAISS поиск |
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

Hybrid retrieval по умолчанию (BM25 + E5 Dense через RRF fusion):

```python
# config.py
RETRIEVAL_MODE = "hybrid"
HYBRID_FUSION = "rrf"
HYBRID_RRF_K = 60
HYBRID_DENSE_WEIGHT = 0.5
```

Cross-encoder re-ranking доступен, но выключен по умолчанию (съедает VRAM вместе с Gemma):

```python
RERANK_ENABLED = False  # True для включения
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
```

## Пороги

| Параметр | Значение | Назначение |
|----------|----------|------------|
| `RELEVANCE_THRESHOLD` | 0.35 | Минимальный score чанка |
| `CONFIDENCE_THRESHOLD` | 0.55 | Hallucination guard (bi-encoder) |
| `CE_CONFIDENCE_THRESHOLD` | 0.50 | Hallucination guard (cross-encoder) |
| `MAX_CONTEXT_CHUNKS` | 5 | Лимит чанков в контексте |
| `CHUNK_SIZE` | 300 | Токенов E5 на чанк |
| `CHUNK_OVERLAP` | 50 | Токенов overlap |

## Скрипты

| Скрипт | Назначение |
|--------|------------|
| `scripts/build_index.py` | Чанки → эмбеддинги → FAISS + Qdrant |
| `scripts/build_bm25.py` | BM25 индекс из metadata.json |
| `scripts/build_golden.py` | Golden QA dataset |
| `scripts/verify_quality.py` | 31 проверка качества данных |
| `scripts/verify_search.py` | 21 тест-запрос retrieval |
| `scripts/check_consistency.py` | Синхронизация chunks/FAISS/Qdrant/golden |
| `scripts/update_data.py` | Полный пайплайн (скрап → индекс) |
| `scripts/test_lm_studio.py` | Проверка подключения к LM Studio |

```bash
```

