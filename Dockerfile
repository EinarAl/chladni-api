FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY chladni ./chladni
COPY app ./app

RUN pip install --no-cache-dir .

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('PORT', '8000'), timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
