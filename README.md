# MediBot — Advanced RAG with Role-Based Access Control

Internal assistant for MediAssist Health Network. Hybrid retrieval (dense + BM25),
cross-encoder reranking, SQL RAG, and RBAC enforced at the Qdrant retrieval layer.

> Build in progress. Sections marked TODO are filled in as components land.

## Setup

```bash
git clone <repo> && cd medibot
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # add your Groq key
python backend/scripts/check_setup.py
```

### Qdrant: pick a mode in `.env`

| `QDRANT_MODE` | What it does | When to use |
|---|---|---|
| `embedded` | Runs inside the Python process, persists to `./qdrant_storage` | Default. No Docker, no server, nothing to install |
| `cloud` | Qdrant Cloud free tier (1GB, no card) | Recommended for the demo — works from any machine |
| `server` | Your own container: `docker compose up -d` | If you already have Docker |

All three run identical application code — only the client constructor differs.

**Embedded caveat:** only one process may hold the storage folder at a time.
Stop the API server before running `ingest.py`, and vice versa.

## Demo credentials

| Username | Password | Role | Can access |
|---|---|---|---|
| dr.mehta | doctor123 | doctor | general, clinical, nursing |
| nurse.priya | nurse123 | nurse | general, nursing |
| billing.ravi | billing123 | billing_executive | general, billing (+ SQL RAG) |
| tech.anand | tech123 | technician | general, equipment |
| admin.sys | admin123 | admin | all (+ SQL RAG) |


## Ingestion

```bash
python backend/scripts/ingest.py --dry-run   # parse & inspect, no indexing
python backend/scripts/ingest.py             # parse & index
python backend/scripts/ingest.py --recreate  # wipe and rebuild
```

## RBAC verification

```bash
python backend/scripts/test_rbac.py
```

Runs 5 adversarial prompts against the live index and reports whether any
restricted collection appeared in the retrieved chunks.

## Retrieval comparison

```bash
python backend/scripts/compare_retrieval.py
python backend/scripts/compare_retrieval.py --role doctor --q "amoxicillin dosage"
```

Prints dense-only, BM25-only, hybrid and hybrid+rerank side by side, plus a
rank-movement table showing how far each chunk moved during reranking.

## Running the API

```bash
cd backend
uvicorn app.api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

**Embedded Qdrant holds a lock** — stop the server before running `ingest.py`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, index readiness, chunk counts per collection |
| `POST /login` | Username + password to a signed JWT carrying the role |
| `POST /chat` | Main RAG endpoint (bearer token required) |
| `GET /collections/{role}` | Collections a role may access |

The role used for filtering comes from the **signed token**, never the
request body — otherwise any client could claim to be admin.
