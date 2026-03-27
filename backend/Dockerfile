FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libheif1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY backend/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY backend/ /app/
COPY frontend/ /app/frontend/
COPY LICENSE /app/LICENSE

ENV DB_PATH=/app/data/photos.db
ENV TORCH_HOME=/app/.cache/torch

RUN mkdir -p /app/data /app/.cache/torch

EXPOSE 8000

# Allow time for CLIP model load on first start before marking unhealthy
HEALTHCHECK --interval=35s --timeout=8s --start-period=120s --retries=4 \
  CMD python /app/healthcheck.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
