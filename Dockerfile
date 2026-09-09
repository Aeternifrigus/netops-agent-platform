FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY app/ app/
COPY data/ data/

# Train at build time so startup doesn't need to.
RUN python -m app.nn.train_anomaly_model

RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
