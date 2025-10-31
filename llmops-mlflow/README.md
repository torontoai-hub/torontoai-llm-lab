# LLM Ops Workshop (Local Stack)

## Quickstart
```bash
cp .env.example .env
docker compose up -d --build
# UIs:
# - LLM API: http://localhost:8000/docs
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
# - MLflow: http://localhost:5000

# Send a test request
curl -s -X POST http://localhost:8000/chat -H "content-type: application/json" -d '{"prompt":"Say hello in 5 words."}' | jq

# Run evaluation
docker compose exec mlflow bash -lc "python eval/eval_with_judge_mlflow_ollama.py"
```
