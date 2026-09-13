"""Component 4: SQL RAG.

The brief asks for a plain Python function with three explicit steps:

    1. natural language -> SQL, via the LLM
    2. clean the raw LLM output down to just the SQL statement
    3. execute, then send the rows back to the LLM for a written answer

Step 2 is the one people skip, and it's the one that breaks. LLMs wrap SQL in
markdown fences, prefix it with "Here's the query:", or append an explanation.
Passing that straight to sqlite3 throws a syntax error.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any, List, Optional, Tuple

from app.core.config import settings

log = logging.getLogger(__name__)

# Only these verbs may ever run. A read-only assistant has no business
# issuing DROP, DELETE or UPDATE — and an LLM that has been prompt-injected
# absolutely should not be able to.
ALLOWED_PREFIXES = ("select", "with")
FORBIDDEN = re.compile(
    r"\b(drop|delete|update|insert|alter|create|truncate|attach|pragma)\b",
    re.IGNORECASE,
)


def get_schema(db_path: Optional[str] = None) -> str:
    """Read the real schema from the database.

    Always read it rather than hardcoding. The LLM writes far better SQL when
    it sees actual column names and sample values, and sample rows are what
    stop it guessing that a status column holds "Escalated" when it really
    holds "escalated".
    """
    path = db_path or settings.sqlite_path
    lines: List[str] = []

    with sqlite3.connect(path) as conn:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_desc = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            lines.append(f"TABLE {table} ({col_desc})")

            # Distinct values for low-cardinality text columns — this is the
            # single most useful thing you can give the model.
            for col in cols:
                name, dtype = col[1], (col[2] or "").upper()
                if "CHAR" in dtype or "TEXT" in dtype:
                    distinct = conn.execute(
                        f"SELECT DISTINCT {name} FROM {table} "
                        f"WHERE {name} IS NOT NULL LIMIT 12"
                    ).fetchall()
                    values = [str(d[0]) for d in distinct]
                    if 0 < len(values) <= 12:
                        lines.append(f"  {name} values: {values}")

            sample = conn.execute(f"SELECT * FROM {table} LIMIT 2").fetchall()
            for row in sample:
                lines.append(f"  example row: {row}")
            lines.append("")

    return "\n".join(lines)


def clean_sql(raw: str) -> str:
    """STEP 2: strip everything that isn't the SQL statement.

    Handles: markdown fences, "Here is the query:" preambles, trailing
    explanations, and stray semicolon spacing.
    """
    text = raw.strip()

    # ```sql ... ``` or ``` ... ```
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    # Drop any leading prose before the first SELECT/WITH
    match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if match:
        text = text[match.start():]

    # Cut anything after the first statement terminator
    if ";" in text:
        text = text.split(";")[0]

    return " ".join(text.split()).strip()


def is_safe(sql: str) -> Tuple[bool, str]:
    """Reject anything that isn't a read."""
    if not sql:
        return False, "No SQL statement could be extracted."
    if not sql.lower().startswith(ALLOWED_PREFIXES):
        return False, "Only SELECT queries are permitted."
    if FORBIDDEN.search(sql):
        return False, "Query contains a forbidden write operation."
    return True, ""


def run_sql(sql: str, db_path: Optional[str] = None) -> Tuple[bool, Any]:
    """STEP 3a: execute against SQLite, read-only, with a row cap."""
    path = db_path or settings.sqlite_path
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(200)
        return True, {"columns": columns, "rows": rows}
    except sqlite3.Error as err:
        return False, f"SQL error: {err}"


SQL_SYSTEM = """You write SQLite queries. You are given a schema with real column names, the distinct values of text columns, and example rows.

Return ONLY the SQL statement. No markdown fences, no explanation, no preamble.

Rules:
- SELECT queries only.
- Use exactly the column names and the exact value casing shown in the schema.
- Dates are stored as TEXT in ISO format; use date() and strftime() for date logic.
- When the question asks "how many", return a COUNT.
- When grouping, always include the grouping column in the SELECT so the result is readable.
- If the question cannot be answered from these tables, return: SELECT 'unanswerable' AS note"""

EXPLAIN_SYSTEM = """You turn SQL results into a short, plain answer for hospital operations staff.

- Lead with the number or finding. One or two sentences.
- Use the exact figures returned. Never estimate or round.
- If the result is empty, say no matching records were found.
- Do not show the SQL unless asked."""


def sql_rag_chain(question: str, api_key: Optional[str] = None) -> str:
    """The three-step chain the brief specifies.

    Returns a natural language answer. Every failure path returns a readable
    string rather than raising, so the API layer never needs a try/except.
    """
    from groq import Groq

    key = api_key or settings.groq_api_key
    if not key:
        return "No LLM API key configured."
    if not settings.sqlite_path.exists():
        return f"Database not found at {settings.sqlite_path}."

    client = Groq(api_key=key)

    # --- STEP 1: question -> SQL ---------------------------------------- #
    try:
        schema = get_schema()
    except sqlite3.Error as err:
        return f"Couldn't read the database schema: {err}"

    try:
        first = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.0,  # deterministic: SQL has right answers
            max_tokens=400,
            messages=[
                {"role": "system", "content": SQL_SYSTEM},
                {"role": "user", "content": f"SCHEMA:\n{schema}\n\nQUESTION: {question}\n\nSQL:"},
            ],
        )
        raw_sql = first.choices[0].message.content or ""
    except Exception as err:  # noqa: BLE001
        return f"Couldn't reach the language model: {str(err)[:160]}"

    # --- STEP 2: clean the output --------------------------------------- #
    sql = clean_sql(raw_sql)
    log.info("raw LLM output: %r", raw_sql[:200])
    log.info("cleaned SQL:    %r", sql)

    ok, reason = is_safe(sql)
    if not ok:
        return f"I couldn't build a safe query for that. ({reason})"

    # --- STEP 3: execute, then explain ---------------------------------- #
    ok, result = run_sql(sql)
    if not ok:
        return f"The query failed against the database. ({result})"

    columns, rows = result["columns"], result["rows"]
    if not rows:
        return "No matching records were found in the database."

    table = f"Columns: {columns}\nRows: {rows[:60]}"
    try:
        second = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.1,
            max_tokens=600,
            messages=[
                {"role": "system", "content": EXPLAIN_SYSTEM},
                {"role": "user", "content": f"QUESTION: {question}\n\nSQL RUN: {sql}\n\nRESULT:\n{table}"},
            ],
        )
        return (second.choices[0].message.content or "").strip() or str(rows[:10])
    except Exception as err:  # noqa: BLE001
        # Still useful: hand back the raw rows rather than nothing.
        return f"Query returned {len(rows)} row(s): {rows[:10]}  (explanation step failed: {str(err)[:80]})"
