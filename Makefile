.PHONY: install train test lint run smoke docker-build clean

install:
	pip install -r requirements.txt

train:
	python -m app.nn.train_anomaly_model

test: train
	pytest -v

lint:
	ruff check app tests scripts

run: train
	uvicorn app.main:app --reload --port 8080

smoke: train
	python scripts/smoke.py

docker-build:
	docker build -t netops-agent-platform .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
