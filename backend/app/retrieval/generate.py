"""Answer generation: turn reranked chunks into a cited answer.

The prompt here does one job above all others: keep the model inside the
supplied context. In a medical setting a fluent invention is worse than
"I don't know" — so the system prompt makes refusal an explicitly correct
outcome rather than a failure.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.core.config import settings
from app.retrieval.hybrid import Hit
from app.retrieval.rerank import build_context

log = logging.getLogger(__name__)

ANSWER_SYSTEM = """You are MediBot, an internal assistant for MediAssist Health Network staff.

You answer ONLY from the numbered context passages provided. They have already been filtered to what this member of staff is authorised to read.

RULES:
1. Use only the supplied passages. If they do not contain the answer, say so plainly: "I couldn't find that in the documents available to you." Do not fall back on general knowledge — a plausible invention is far more dangerous here than an admission.
2. Cite the passage number inline, like [1] or [2], for every factual claim.
3. Preserve exact values precisely: dosages, ICD codes, prices, model numbers, timeframes. Never round, never approximate, never convert units.
4. If passages disagree, say so and cite both rather than silently picking one.
5. Never speculate about documents you cannot see, and never mention that other collections exist.
6. Be concise and practical. Staff are working — lead with the answer, then the detail.
7. You provide reference information from hospital documents. You do not give clinical advice, and you never override a clinician's judgement."""


def answer_from_context(
    question: str,
    hits: List[Hit],
    api_key: Optional[str] = None,
) -> Tuple[bool, str]:
    """Generate a cited answer from the reranked chunks. Never raises."""
    if not hits:
        return True, (
            "I couldn't find anything relevant in the documents available to you. "
            "Try rephrasing, or check whether this sits in a collection your role "
            "can access."
        )

    from groq import Groq

    key = api_key or settings.groq_api_key
    if not key:
        return False, "No LLM API key configured. Set GROQ_API_KEY in .env."

    user_prompt = (
        f"CONTEXT PASSAGES:\n\n{build_context(hits)}\n\n"
        f"{'=' * 60}\n\nSTAFF QUESTION: {question}\n\n"
        "Answer using only the passages above, citing them by number."
    )

    try:
        response = Groq(api_key=key).chat.completions.create(
            model=settings.llm_model,
            temperature=0.2,  # low: we want faithful, not creative
            max_tokens=1200,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as err:  # noqa: BLE001
        log.error("LLM call failed: %s", err)
        return False, f"Couldn't reach the language model: {str(err)[:180]}"

    if not text:
        return False, "The model returned an empty response. Please try again."
    return True, text


# --------------------------------------------------------------------------- #
# Routing: documents or database?
# --------------------------------------------------------------------------- #

# Words that suggest the answer lives in a table of rows, not a PDF.
ANALYTICAL_MARKERS = [
    "how many", "how much", "count", "total", "sum", "average", "mean",
    "number of", "most", "least", "highest", "lowest", "top ", "breakdown",
    "per department", "by department", "by category", "by status",
    "last month", "last quarter", "this year", "trend", "statistics",
    "open tickets", "pending claims", "escalated", "rejected",
]


def is_analytical(question: str) -> bool:
    """Cheap keyword router: does this look like a database question?

    A keyword check is deliberate. An LLM classifier would cost a round-trip
    on every single request, and the failure mode is mild — a misrouted
    question simply searches documents and finds nothing useful, rather than
    leaking anything.
    """
    low = question.lower()
    return any(marker in low for marker in ANALYTICAL_MARKERS)
