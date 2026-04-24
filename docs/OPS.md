# Erik Operations Guide

## Production architecture (post-2026-04-24 split)

```
┌─ Vercel ─────────────────────────────┐
│  erik-website-eosin                   │  Next.js, family-facing
└──────────────┬────────────────────────┘
               │ HTTPS
┌──────────────▼────────────────────────┐   ┌────────────────────────────┐
│  erik-api (Railway)                    │   │  erik-research-worker      │
│  FastAPI /api/* + /health              │   │  (Railway)                 │
│  ERIK_RESEARCH_LOOP=false              │   │  run_loop + /health/...    │
│  Healthcheck: /health                  │   │  ERIK_RESEARCH_LOOP=true   │
│  erik-api-production.up.railway.app    │   │  ERIK_SKIP_MIGRATIONS=true │
└──────────────┬────────────────────────┘   │  Healthcheck: /health/     │
               │                             │              research       │
               │                             └─────────────┬──────────────┘
               │                                           │
               └───────────────┬───────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Railway Postgres (shared)      │
              │  erik_core, erik_ops schemas    │
              └─────────────────────────────────┘
```

## Invariants

* **Only one service runs the research loop at a time.** The `SCMWriter`
  advisory lock (`ERIK_SCM_WRITER_LOCK_KEY=7_020_426_101`) prevents two
  Python processes from acquiring it concurrently. If the worker goes
  down and the api is flipped to `ERIK_RESEARCH_LOOP=true`, the api will
  pick up the lock on restart. The opposite transfer also works.

* **Only erik-api is family-facing.** Vercel's `NEXT_PUBLIC_API_URL`
  points at `erik-api-production.up.railway.app`. The worker has a
  Railway-provided public URL but nothing should be sending traffic to
  it. The worker's public URL exists only because Railway's CLI
  auto-generates one on first query.

* **Both services answer /health and /health/research.** This is
  intentional — it lets ops poll either surface without caring which
  service is running the loop.

## Per-service config that lives in the Railway UI

`railway.toml` applies globally. Per-service overrides must be set in
the Railway UI (Settings → Deploy) for each service:

* **erik-api**: Healthcheck Path = `/health`
* **erik-research-worker**: Healthcheck Path = `/health/research`

Without this, Railway's auto-restart will only catch "process died"
failures. With `/health/research`, it also catches "research loop hung
but process still alive" — which is the silent-failure mode that an
inside-the-api loop had no defense against.

## Env vars (per service)

### Shared (same on both)
* `ANTHROPIC_API_KEY` — Claude API
* `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` — Bedrock
* `DATABASE_URL` — Railway PG (internal URL)
* `ERIK_INVITE_CODES` — comma-separated family invite codes
* `ERIK_LLM_BACKEND=bedrock`
* `ERIK_LLM_MONTHLY_BUDGET_USD` — hard ceiling (default $150)
* `CONNECTOR_MODE=api`
* `PYTHONUNBUFFERED=1`

### erik-api only
* `ERIK_RESEARCH_LOOP=false`
* `CORS_ALLOWED_ORIGINS` — Vercel origins

### erik-research-worker only
* `ERIK_RESEARCH_LOOP=true`
* `ERIK_SKIP_MIGRATIONS=true` — api has already run them; prevents
  concurrent migration runs that can deadlock under load.

## Activation flags in `data/erik_config.json`

The SCM + CPTS + R4 stack follows Galen's substrate-active/consumer-off
discipline. Current state after 2026-04-24:

| Flag | Current | Flip after… |
|---|---|---|
| `scm_split_enabled` | `true` | — (live) |
| `scm_writer_enabled` | `true` | — (live) |
| `scm_bootstrap_enabled` | `true` | — (live, populating) |
| `effect_enricher_enabled` | `false` | bootstrap has promoted ≥10 compound→gene edges |
| `r4_propagation_enabled` | `false` | effect_enricher has been live 5+ days with clean telemetry |
| `cpts_enabled` | `false` | r4 has been live 1 week without cascading refutations |

See the PR description for `feat(scm+cpts+r4)` for the full progressive
activation plan.

## Smoke tests

From the command line:

```
curl -s https://erik-api-production.up.railway.app/health
curl -s https://erik-api-production.up.railway.app/health/research
# /api/summary without a cookie → expect 401
curl -s -o /dev/null -w "%{http_code}\n" \
    https://erik-api-production.up.railway.app/api/summary
```

## Stopping / restarting the research loop

The worker is stateless between cycles — everything lives in PG. Safe
operations:

* **Pause research**: `railway variables --service erik-research-worker
  --set ERIK_RESEARCH_LOOP=false`, then `railway redeploy --service
  erik-research-worker`.
* **Resume**: flip back to `true`, redeploy.
* **Emergency stop all LLM calls**: `railway variables --service
  erik-research-worker --set ERIK_LLM_MONTHLY_BUDGET_USD=0`. The
  spend gate refuses every call immediately; the loop keeps running
  (DB ops only) but costs nothing.

## Rollback R4 propagation events

If R4 fires on a scm_edge that turns out to have been validly active:

```
PYTHONPATH=scripts python -m ops.propagation_rollback list --limit 20
PYTHONPATH=scripts python -m ops.propagation_rollback show <event_id>
PYTHONPATH=scripts python -m ops.propagation_rollback rollback <event_id>
```

Against Railway PG:
```
DATABASE_URL=<railway-pg-url> PYTHONPATH=scripts \
    python -m ops.propagation_rollback rollback <event_id>
```
