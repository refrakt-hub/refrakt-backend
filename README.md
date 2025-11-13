## Runtime Operations

- **Lifespan Startup**: `main.py` initialises Prometheus metrics, queue monitor, and the Redis-backed rate limiter during application startup. Ensure Redis (same URL as `QUEUE_URL`) is reachable before launching the API.
- **Shutdown**: Lifespan teardown stops the monitor and closes limiter connections; graceful termination is required for clean Redis disconnects.

## Observability

- **Metrics Endpoint**: FastAPI exposes Prometheus metrics at `PROMETHEUS_METRICS_ROUTE` (default `/metrics`). Protect it with `PROMETHEUS_BEARER_TOKEN` or upstream ACLs (Cloudflare Zero Trust).  
- **Sampling Interval**: Queue gauges update every `PROMETHEUS_QUEUE_POLL_INTERVAL` seconds (default 5). Increase if Redis pressure is observed.
- **Key Metrics**:
  - `refrakt_job_enqueued_total{source=...}`
  - `refrakt_job_status_transitions_total{from_status,to_status}`
  - `refrakt_job_queue_wait_seconds` (histogram)
  - `refrakt_queue_pending_jobs`, `refrakt_queue_running_jobs`, `refrakt_queue_oldest_pending_seconds`

### Prometheus Scrape Template

Use this example when deploying Prometheus (adjust host, token, and scrape interval to match your tunnel):

```
scrape_configs:
  - job_name: refrakt-backend
    scrape_interval: 5s
    metrics_path: /metrics             # or value of PROMETHEUS_METRICS_ROUTE
    scheme: https                      # when scraping via Cloudflare tunnel
    authorization:
      credentials: Bearer ${PROM_BEARER_TOKEN}
    static_configs:
      - targets:
          - api.dev.akshath.tech       # tunnel hostname or internal IP
```

Store `${PROM_BEARER_TOKEN}` in your secrets manager and inject it into Prometheus’ environment or configuration file at runtime.

## Rate Limiting & Admission Control

- **Library**: `fastapi-limiter` uses the same Redis instance as SAQ (`QUEUE_URL`).  
- **Endpoints Covered**: `/run` and `/assistant`. Other endpoints remain unrestricted.
- **Identifiers**:
  - Primary: client IP resolved via `CLIENT_IP_HEADER` (`CF-Connecting-IP` by default) then `X-Forwarded-For` fallback.
  - Secondary: user identity derived from `X-User-ID` (or `X-UserID`, or `user_id` query param). Anonymous traffic falls back to IP.
- **Tiering**: Clients can send `X-RateLimit-Tier: premium` to receive `RATE_LIMIT_PREMIUM_MULTIPLIER` × allowances (must be authorised upstream).
- **Quotas (per 60s window)**:
  - `RATE_LIMIT_RUN_BURST` and `RATE_LIMIT_RUN_PER_MINUTE`
  - `RATE_LIMIT_ASSISTANT_PER_MINUTE`
- **Responses**:
  - 429 Too Many Requests → rate-limiter exhaustion (FastAPI-Limiter default JSON).
  - 503 Service Unavailable + `Retry-After` → queue back-pressure triggered (`QUEUE_MAX_PENDING` or `QUEUE_MAX_AGE_SECONDS` exceeded).

## Environment Variables

- Observability: `PROMETHEUS_ENABLED`, `PROMETHEUS_METRICS_ROUTE`, `PROMETHEUS_BEARER_TOKEN`, `PROMETHEUS_QUEUE_POLL_INTERVAL`
- Rate Limiting: `RATE_LIMIT_ENABLED`, `RATE_LIMIT_RUN_BURST`, `RATE_LIMIT_RUN_PER_MINUTE`, `RATE_LIMIT_ASSISTANT_PER_MINUTE`, `RATE_LIMIT_PREMIUM_MULTIPLIER`, `CLIENT_IP_HEADER`
- Admission Control: `QUEUE_MAX_PENDING`, `QUEUE_MAX_AGE_SECONDS`

Populate sensitive values through your secrets manager (planned in the upcoming “Security & Secrets Management” milestone). Commit only example defaults to source control.

## Validation Checklist

- **Smoke Tests**:
  - Launch API with Redis and hit `/metrics` using the bearer token; confirm gauges update after a sample job run.
  - Submit > `RATE_LIMIT_RUN_BURST` requests from same IP; expect 429 with limiter JSON body.
  - Enqueue enough jobs to push `QUEUE_MAX_PENDING` or `QUEUE_MAX_AGE_SECONDS`; expect 503 with `Retry-After` header.
- **Load Tests**:
  - Use Locust/Vegeta to simulate 50 concurrent users hitting `/run` & `/assistant`.
  - Track Prometheus metrics for queue depth, wait time, limiter rejections; verify thresholds produce alerts.
- **Monitoring Setup**:
  - Deploy Prometheus (binary/Docker) with the provided scrape config.
  - Create dashboard panels for queue gauges and job histograms; set alert rules on `refrakt_queue_pending_jobs` and `refrakt_queue_oldest_pending_seconds`.

Document outcomes and thresholds in the ops runbook after the first full load test cycle.

