# Руководство по развёртыванию МИРЭА RAG Bot

## Что это

RAG-система (Retrieval-Augmented Generation) для ответов на вопросы о РТУ МИРЭА.
Включает веб-интерфейс (чат), API и CLI. Работает полностью локально, без облачных сервисов.

**Состав:**
- Векторная база Qdrant (1459 чанков — данные с mirea.ru + QA-база ВИКА)
- Эмбеддинги E5-small (384d, локально)
- LLM через LM Studio (Gemma 4 E2B, ~5 ГБ)
- Веб-интерфейс на localhost:8080

---

## Требования к системе

| Компонент | Минимум |
|-----------|---------|
| ОС | Windows 10/11, Linux, macOS |
| RAM | 8 ГБ (лучше 16 ГБ) |
| VRAM (видеокарта) | 6 ГБ (для LLM), или CPU (медленнее) |
| Место на диске | ~2 ГБ (проект) + ~5 ГБ (модель LLM) |
| **LM Studio** | [lmstudio.ai](https://lmstudio.ai) — LLM сервер |
| **Docker** | Для запуска RAG API (рекомендуется) |

---

## Шаг 1. Установка LM Studio и модели

LM Studio — программа для запуска LLM локально. Работает на хосте (вне Docker).

### 1.1. Установка

1. Скачайте LM Studio с [lmstudio.ai](https://lmstudio.ai)
2. Установите и запустите

### 1.2. Загрузка модели

1. Откройте LM Studio → вкладка **Search** (значок лупы)
2. Введите в поиск: `gemma-4-e2b`
3. Найдите модель от Google (формат GGUF)
4. Скачайте версию **Q4_K_M** (~2–3 ГБ)

### 1.3. Запуск сервера

1. Перейдите на вкладку **Developer** (значок </>)
2. Выберите загруженную модель из выпадающего списка
3. Нажмите **Start Server**
4. Порт: **1234** (по умолчанию)
5. Убедитесь, что горит зелёный индикатор «Server started»

> **Важно:** LM Studio должен работать всё время, пока используется бот.

---

## Шаг 2. Установка Docker

### Windows / macOS

1. Установите [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Запустите Docker Desktop
3. Проверьте: `docker --version` и `docker compose version`

### Linux

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# Выйти и зайти снова
```

---

## Шаг 3. Запуск (Docker — рекомендуется)

1. Распакуйте архив `mirea-rag-bot.zip` в удобное место

2. Откройте терминал в папке проекта и запустите:

```bash
docker compose up --build -d
```

3. Дождитесь сборки и запуска (первый раз ~3–5 минут):
   - Скачиваются Python-зависимости
   - Загружается модель E5-small (~120 МБ из HuggingFace)
   - Инициализируется Qdrant

4. Проверьте:

```bash
docker compose logs -f
```

Должно появиться: `Uvicorn running on http://0.0.0.0:8080`

5. Откройте в браузере: **http://localhost:8080**

Готово! Можно задавать вопросы.

### Остановка

```bash
docker compose down
```

### Перезапуск

```bash
docker compose up -d
```

---

## Альтернатива: запуск без Docker (ручная установка)

Если Docker недоступен, можно запустить напрямую через Python.

### 1. Установка Python

1. Установите Python 3.11+ с [python.org](https://www.python.org/downloads/)
2. При установке отметьте «Add Python to PATH»

### 2. Установка зависимостей

```bash
cd mirea-rag-bot
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Запуск

```bash
python -m uvicorn src.rag.api:app --host 0.0.0.0 --port 8080
```

---

## Возможные проблемы

### LM Studio не подключается (из Docker)

1. Убедитесь, что LM Studio сервер запущен (зелёный индикатор)
2. Проверьте порт: **1234**
3. В LM Studio → Developer → включите **CORS** и **Serve locally**
4. Проверьте из контейнера:

```bash
docker compose exec rag-api curl http://host.docker.internal:1234/v1/models
```

### Модель E5 не скачивается (нет интернета в контейнере)

Модель скачается автоматически (~120 МБ). Если не работает:

1. Скачайте вручную: [huggingface.co/intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
2. Распакуйте в `models/e5-small/` внутри проекта
3. В `config.py` укажите: `EMBEDDING_MODEL = "models/e5-small"`

### Память (OOM / тормоза)

- В LM Studio включите **CPU offload** если мало VRAM
- Уменьшите `MAX_CONTEXT_CHUNKS` в `src/rag/answer.py` (строка 86) с 5 до 3

### Порт 8080 занят

Измените в `docker-compose.yml`:
```yaml
ports:
  - "9000:8080"   # внешний порт : внутренний
```

---

## Структура проекта

```
mirea-rag-bot/
├── Dockerfile              ← образ Docker
├── docker-compose.yml      ← оркестрация
├── config.py               ← настройки
├── requirements.txt        ← Python-зависимости
├── INSTALL.md              ← этот файл
├── src/
│   ├── rag/                ← RAG-пайплайн (answer, api, retriever)
│   ├── scraper/            ← скраперы mirea.ru
│   ├── retrieval/          ← гибридный поиск
│   └── pipeline/           ← очистка данных
├── static/                 ← веб-интерфейс
├── scripts/                ← утилиты (оценка, индексация)
├── tests/                  ← тесты
└── mirea_data/             ← данные (примонтированы как volume)
    ├── qdrant/             ← векторная база (готова!)
    ├── bm25/               ← BM25 индекс (готов!)
    ├── faiss/metadata.json ← все чанки
    └── eval/               ← golden set
```

---

## API

| URL | Метод | Описание |
|-----|-------|----------|
| `/` | GET | Веб-интерфейс (чат) |
| `/ask` | POST | `{question, history?}` → ответ |
| `/health` | GET | Статус: E5, Qdrant, LM Studio |
| `/analytics` | GET | Статистика запросов |

### Пример

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Где находится главный корпус МИРЭА?"}'
```

---

## Конфигурация

Основные настройки — в `config.py`:

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `TOP_K_RETRIEVAL` | 3 | Чанков на поиск |
| `MAX_CONTEXT_CHUNKS` | 5 | Чанков в контексте LLM |
| `RETRIEVAL_MODE` | "hybrid" | "hybrid" (BM25+Dense) или "dense" |
| `LM_STUDIO_URL` | env / localhost:1234 | URL сервера LM Studio |

LM Studio URL настраивается через переменную окружения `LM_STUDIO_URL` в `docker-compose.yml`.
