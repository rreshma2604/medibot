"""Component 3: cross-encoder reranking.

Bi-encoder (what retrieval uses):
    embed(query) and embed(chunk) are computed SEPARATELY, then compared.
    The chunk's vector was computed at ingestion time, long before anyone
    asked this question. Fast — you can compare against millions.

Cross-encoder (what this module uses):
    the model reads "query [SEP] chunk" as ONE input and outputs a single
    relevance score. Its attention layers can relate specific query words to
    specific chunk words. Far more accurate, far slower — you can only afford
    it on a handful of candidates.

Hence the funnel: bi-encoder narrows millions to 10, cross-encoder narrows
10 to 3. Neither could do the other's job.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional

from app.core.config import settings
from app.retrieval.hybrid import Hit

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cross_encoder():
    """Load the cross-encoder once. First call downloads ~90MB."""
    from sentence_transformers import CrossEncoder

    log.info("loading reranker %s", settings.rerank_model)
    return CrossEncoder(settings.rerank_model)


def rerank(
    question: str,
    candidates: List[Hit],
    top_k: Optional[int] = None,
    log_scores: bool = False,
) -> List[Hit]:
    """Score each candidate against the query jointly, return the best few.

    The brief is explicit that only the reranked top chunks may reach the
    LLM, so this returns a SHORTER list — it does not merely reorder.
    """
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []

    model = _cross_encoder()

    # Pair the query with every candidate. The model sees both together.
    pairs = [(question, hit.text) for hit in candidates]
    scores = model.predict(pairs)

    for hit, score in zip(candidates, scores):
        hit.rerank_score = float(score)

    ranked = sorted(candidates, key=lambda h: h.rerank_score or 0.0, reverse=True)

    if log_scores:
        _log_movement(candidates, ranked, top_k)

    return ranked[:top_k]


def _log_movement(original: List[Hit], ranked: List[Hit], top_k: int) -> None:
    """Print how far each chunk moved. This is the tip from the brief.

    You will often see the chunk that hybrid search ranked 5th score highest
    once the cross-encoder actually reads it against the query. Seeing that
    in your own output is what makes reranking click.
    """
    position = {id(hit): i for i, hit in enumerate(original)}
    print(f"\n  {'was':>4} {'now':>4} {'rerank':>9}  {'kept':>5}  chunk")
    print("  " + "-" * 74)
    for new_index, hit in enumerate(ranked):
        old_index = position[id(hit)]
        moved = old_index - new_index
        arrow = f"+{moved}" if moved > 0 else (str(moved) if moved < 0 else "=")
        kept = "yes" if new_index < top_k else ""
        snippet = hit.text.replace("\n", " ")[:44]
        print(
            f"  {old_index + 1:>4} {new_index + 1:>4} {hit.rerank_score:>9.3f}"
            f"  {kept:>5}  [{arrow:>3}] {snippet}"
        )
    print()


def build_context(hits: List[Hit]) -> str:
    """Format the surviving chunks for the LLM prompt.

    Each chunk is numbered and labelled with its source so the model can
    cite by number, and so a human reading the prompt can audit what the
    model was given.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(
            f"[{i}] Source: {hit.source_document}"
            f" | Section: {hit.section_title}"
            f" | Collection: {hit.collection}\n{hit.text}"
        )
    return "\n\n---\n\n".join(blocks)
