.PHONY: install train test test-tenancy lint run migrate dev-db dev-redis worker smoke docker-build clean

install:
	pip install -r requirements.txt

train:
	python -m app.nn.train_anomaly_model

# Without DATABASE_URL the tenancy tests are skipped.
test: train
	pytest -v

dev-db:
	docker run -d --name netops-db -p 5432:5432 \
		-e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=netops postgres:16
	sleep 5
	docker exec netops-db psql -U postgres -d netops -c \
		"CREATE ROLE app LOGIN PASSWORD 'app' NOSUPERUSER NOBYPASSRLS NOCREATEROLE; \
		 ALTER DATABASE netops OWNER TO app; GRANT ALL ON SCHEMA public TO app;"

migrate:
	alembic upgrade head

dev-redis:
	docker run -d --name netops-redis -p 6379:6379 redis:7

# Needs DATABASE_URL and BROKER_URL.
worker:
	celery -A app.tasks:celery_app worker --loglevel=info

test-tenancy: train migrate
	pytest -v

lint:
	ruff check app tests scripts alembic

run: train
	uvicorn app.main:app --reload --port 8080

smoke: train
	python scripts/smoke.py

docker-build:
	docker build -t netops-agent-platform .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
