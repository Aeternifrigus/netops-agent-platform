FROM python:3.12-slim AS builder
WORKDIR /build
# A virtualenv at a world-readable path. `pip install --user` would put the
# packages under /root/.local, which the non-root runtime user cannot read.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY app/ app/
COPY data/ data/
# Migrations ship in the image so the Kubernetes migration Job can run them.
COPY alembic/ alembic/
COPY alembic.ini .

# Train at build time so startup doesn't need to.
RUN python -m app.nn.train_anomaly_model

RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
