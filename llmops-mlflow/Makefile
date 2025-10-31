.PHONY: up down logs pull eval ui grafana creds

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200 llm-service

pull:
	docker compose pull

eval:
	docker compose exec mlflow bash -lc "python eval/eval_with_judge_mlflow_ollama.py"

ui:
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "MLflow: http://localhost:5000"
	@echo "LLM API: http://localhost:8000/docs"

grafana:
	open http://localhost:3000 || true

creds:
	@echo "Grafana admin / admin"
