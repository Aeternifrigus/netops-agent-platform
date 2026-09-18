.PHONY: install train test test-tenancy lint run migrate dev-db smoke docker-build clean

install:
	pip install -r requirements.txt

train:
	python -m app.nn.train_anomaly_model

# Runs in open mode: the tenancy tests skip, everything else runs, and no
# database is needed. This is the bare-checkout path.
test: train
	pytest -v

# A local PostgreSQL with the unprivileged application role already created.
dev-db:
	docker run -d --name netops-db -p 5432:5432 \
		-e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=netops postgres:16
	sleep 5
	docker exec netops-db psql -U postgres -d netops -c \
		"CREATE ROLE app LOGIN PASSWORD 'app' NOSUPERUSER NOBYPASSRLS NOCREATEROLE; \
		 ALTER DATABASE netops OWNER TO app; GRANT ALL ON SCHEMA public TO app;"

migrate:
	alembic upgrade head

# The full suite, including tenant isolation against a real PostgreSQL.
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
