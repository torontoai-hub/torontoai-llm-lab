FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SERVICE_NAME=llm-service     OPENAI_BASE_URL=http://ollama:11434/v1     OPENAI_API_KEY=ollama     MODEL_NAME=llama3.2     MODEL_TEMPERATURE=0.2     OTLP_ENDPOINT=http://tempo:4318/v1/traces

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
