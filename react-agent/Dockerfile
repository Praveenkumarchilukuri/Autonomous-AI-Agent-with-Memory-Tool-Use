FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/faiss_store /app/evaluation/results

EXPOSE 8000 7860

# Default: FastAPI — override with `docker-compose` for Gradio
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
