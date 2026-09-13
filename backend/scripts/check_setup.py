"""Run this FIRST, before writing any pipeline code.

It checks every moving part independently so that when something breaks
later you already know which pieces were working. Debugging a six-component
system is mostly about narrowing down where the failure lives.

    python backend/scripts/check_setup.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core import rbac  # noqa: E402

OK, BAD, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"
failures = 0


def report(passed: bool, message: str, fatal: bool = True) -> None:
    global failures
    print(f"{OK if passed else (BAD if fatal else WARN)} {message}")
    if not passed and fatal:
        failures += 1


print("\n=== 1. Python packages ===")
for module, label in [
    ("qdrant_client", "qdrant-client"),
    ("fastembed", "fastembed (dense + sparse embeddings)"),
    ("docling", "docling (structural PDF parsing)"),
    ("groq", "groq (LLM API)"),
    ("fastapi", "fastapi"),
    ("sentence_transformers", "sentence-transformers (reranker)"),
]:
    try:
        __import__(module)
        report(True, label)
    except ImportError:
        report(False, f"{label} — missing, run: pip install -r requirements.txt")


print("\n=== 2. Data files ===")
docs = settings.documents_dir
report(docs.exists(), f"documents folder exists: {docs}")
if docs.exists():
    pdfs = list(docs.rglob("*.pdf")) + list(docs.rglob("*.md"))
    report(len(pdfs) > 0, f"found {len(pdfs)} document(s)")
    subdirs = sorted(p.name for p in docs.iterdir() if p.is_dir())
    print(f"         collection folders: {subdirs or 'none — see note below'}")

report(settings.sqlite_path.exists(), f"database exists: {settings.sqlite_path}")


print("\n=== 3. Database schema ===")
if settings.sqlite_path.exists():
    conn = sqlite3.connect(settings.sqlite_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    report("claims" in tables, f"tables present: {tables}")
    for table in tables:
        cols = [(r[1], r[2]) for r in conn.execute(f"PRAGMA table_info({table})")]
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"\n         {table}  ({count} rows)")
        for name, dtype in cols:
            print(f"           - {name} ({dtype})")
        sample = conn.execute(f"SELECT * FROM {table} LIMIT 2").fetchall()
        for row in sample:
            print(f"           e.g. {row}")
    conn.close()


print("\n=== 4. Qdrant ===")
print(f"         mode: {settings.qdrant_mode}")
try:
    from app.ingestion.store import get_client

    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    where = settings.qdrant_path if settings.qdrant_mode == "embedded" else settings.qdrant_url
    report(True, f"connected ({where}) — collections: {existing or 'none yet'}")
    client.close()
except Exception as err:  # noqa: BLE001
    hint = ("another process may be holding the embedded folder — stop the API server"
            if settings.qdrant_mode == "embedded" else "check the URL and API key")
    report(False, f"Qdrant unreachable: {err} — {hint}")


print("\n=== 5. LLM API ===")
if not settings.groq_api_key:
    report(False, "GROQ_API_KEY not set in .env")
else:
    try:
        from groq import Groq

        Groq(api_key=settings.groq_api_key).chat.completions.create(
            model=settings.llm_model,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        report(True, f"{settings.llm_model} responding")
    except Exception as err:  # noqa: BLE001
        report(False, f"LLM call failed: {str(err)[:160]}")


print("\n=== 6. RBAC matrix sanity ===")
report(rbac.collections_for("nurse") == ["general", "nursing"],
       f"nurse sees {rbac.collections_for('nurse')}")
report(len(rbac.collections_for("admin")) == 5,
       f"admin sees all {len(rbac.collections_for('admin'))} collections")
report(rbac.roles_for("billing") == ["admin", "billing_executive"],
       f"billing readable by {rbac.roles_for('billing')}")
report(not rbac.can_access("nurse", "billing"), "nurse blocked from billing")
report(not rbac.can_use_sql("nurse"), "nurse blocked from SQL RAG")
print(f"\n         Sample refusal:\n         \"{rbac.refusal_message('nurse', 'billing')}\"")


print("\n" + "=" * 62)
print("ALL CHECKS PASSED — you're ready to ingest." if not failures
      else f"{failures} check(s) failed. Fix those before continuing.")
print("=" * 62 + "\n")
sys.exit(1 if failures else 0)
