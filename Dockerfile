FROM python:3.11-slim

# System deps for numpy, lxml, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY src/ ./src/
COPY static/ ./static/

# mirea_data is mounted as volume in docker-compose
# (Qdrant DB, BM25 index, metadata — all persistent data)

EXPOSE 8080

CMD ["uvicorn", "src.rag.api:app", "--host", "0.0.0.0", "--port", "8080"]
