# NetOps Agent Platform

Multi-tenant assistant for network operations. Agents answer questions about a
cell-tower network: what goes down if a tower fails, whether a telemetry reading
looks anomalous, and which recent events are relevant to an incident. Tenant
isolation is enforced by PostgreSQL row-level security, not by application code.

## Features

- Multi-agent orchestrator on Google ADK, with plain Python functions as tools
- Topology graph in Neo4j, with an in-memory fallback
- Anomaly scoring with a NumPy MLP (hand-written backprop, gradient-checked)
- Event relevance ranking with a NumPy multi-head attention implementation
- JWT auth (OAuth2 password flow) with viewer / operator / admin roles
- Row-level security on every tenant table, plus an audit log
- Background agent runs on Celery and Redis
- GraphQL read API alongside REST
- Prompt-injection screening on `/chat`

## Quick start

```bash
pip install -r requirements.txt
make train        # trains the anomaly model
make run          # http://localhost:8080/docs
make test
```

Nothing external is required to run it locally. The `make` targets set
`ENVIRONMENT=local` and `ALLOW_OPEN_MODE=true`; outside `make`, copy
`.env.example` to `.env` for the same defaults.

| Not configured | Behaviour |
|---|---|
| `ENVIRONMENT` | treated as `production`: refuses to start without `DATABASE_URL` and a JWT secret of at least 32 characters |
| Neo4j | in-memory graph, loaded read-only from `data/` |
| `DATABASE_URL` | refuses to start, unless `ALLOW_OPEN_MODE=true` and `ENVIRONMENT=local`, which runs open mode: no auth, no tenants |
| `BROKER_URL` | `/chat` runs inline; background requests return 503 |
| `GOOGLE_API_KEY` | `/chat` returns 503; `/tools/*` still work |

### With Postgres and Redis

```bash
make dev-db                     # Postgres with an unprivileged app role
export DATABASE_URL=postgresql+asyncpg://app:app@localhost:5432/netops
make migrate

make dev-redis
export BROKER_URL=redis://localhost:6379/0
make worker                     # in a second terminal

make test-tenancy
```

### Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml
# create netops-secrets from k8s/secret.example.yaml (DATABASE_URL and JWT_SECRET are required)
kubectl apply -f k8s/migrate-job.yaml
kubectl -n netops-agents wait --for=condition=complete job/netops-migrate --timeout=120s
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
```

## API

| Method | Path | Role |
|---|---|---|
| GET | `/health` | none |
| POST | `/tenants`, `/auth/token`, `/auth/refresh` | none |
| GET | `/users/me` | any |
| POST | `/users` | admin |
| POST | `/tools/impact`, `/tools/health-score`, `/tools/relevance` | viewer |
| POST | `/chat` (`"background": true` returns 202 + task id) | operator |
| GET | `/tasks`, `/tasks/{id}` | viewer |
| GET | `/conversations`, `/conversations/{id}/messages`, `/invocations` | viewer |
| GET | `/audit` | admin |
| POST | `/graphql` | viewer (per-field checks) |

Log in with the tenant slug in the OAuth2 `scope` field, since emails are only
unique within a tenant.

## Design notes

**Tenant isolation.** Each tenant table has a policy comparing `tenant_id` with
`app.tenant_id`, set per transaction with `set_config(..., true)` so it can't
leak across pooled connections. Tables use `FORCE ROW LEVEL SECURITY` so the
owner isn't exempt, and the app role has neither `SUPERUSER` nor `BYPASSRLS`.
Handlers never filter by tenant themselves. CI fails if the catalogue doesn't
show every tenant table enforced.

**Background runs.** The queue message carries only ids. The worker claims the
task with `UPDATE ... WHERE status = 'pending' RETURNING prompt` under the
tenant's scope, so a message sent with the wrong tenant finds nothing, and a
redelivered message doesn't run twice. Task state lives in Postgres rather than
the Celery result backend, so a task id is useless to another tenant.

**Prompt injection.** The regex checks only catch obvious attempts. The real
controls are that tool access comes from the caller's role, untrusted content is
wrapped and its delimiters stripped, and refusals are written to the audit log.

**GraphQL.** Read-only, built on the same session and auth dependencies as REST.
Admin-only fields check the role inside the resolver.

## Tests

84 tests. 49 of them need Postgres (and Redis for the background ones) and are
skipped without `DATABASE_URL`. `scripts/smoke.py` exercises the running app in
either mode.

## Layout

```
app/
  agents/      ADK orchestrator and tool functions
  graph/       Neo4j and in-memory topology backends
  nn/          MLP, attention, anomaly model training
  routers/     auth endpoints
  main.py      FastAPI app
  deps.py      auth, tenant scoping, roles, audit
  models.py    SQLAlchemy models
  tasks.py     Celery worker task
  graphql_api.py
  guard.py     prompt-injection checks
alembic/       migrations, including the RLS policies
k8s/           deployment manifests
tests/
```

## Limitations

- No live model run has been tested (no Gemini key), so `/chat` is verified up
  to the model call.
- The Kubernetes manifests pass schema validation (kubeconform) but haven't been
  applied to a real cluster, and the Neo4j backend hasn't been run against a
  real instance. There is no manifest for the Celery worker yet.
- No rate limiting, no refresh-token revocation, no retries for failed background
  runs, and no depth or cost limits on GraphQL queries.
