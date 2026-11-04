FROM python:3.11-slim

WORKDIR /app

ENV PORT=7860 \
    PYTHONUNBUFFERED=1

COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

COPY src/ ./src/
COPY models/ ./models/
COPY app_api.py .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port ${PORT}"]