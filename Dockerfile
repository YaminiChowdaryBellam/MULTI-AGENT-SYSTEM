# Mirrors RAG-medical-system's Dockerfile pattern (slim base, cached deps layer,
# uvicorn CMD) — no build-essential/torch needed here, but Presidio's spaCy
# model has to be fetched at build time same as pip deps.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Copy source code — deliberately not `.env`: baking secrets into an image
# layer means anyone who pulls the image gets the keys. Pass them at runtime
# via docker-compose's env_file (or -e/--env-file on `docker run`) instead.
COPY agents/ ./agents/
COPY tools/ ./tools/
COPY graph/ ./graph/
COPY api.py .
COPY main.py .

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
