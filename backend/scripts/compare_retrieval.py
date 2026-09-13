"""Compare dense-only vs BM25-only vs hybrid vs hybrid+rerank.

    python backend/scripts/compare_retrieval.py
    python backend/scripts/compare_retrieval.py --role doctor --q "amoxicillin dosing"

The brief asks you to show hybrid retrieval is "demonstrably better than
dense-only". This prints that comparison side by side so you can screenshot
it for the README.

What to look for: queries containing exact medical terms (drug names, ICD
codes, model numbers) are where dense-only fails and BM25 rescues it. Purely
conceptual queries are where the reverse happens. Hybrid should be at least
as good as the better of the two on every query — that's the whole point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.ingestion import store  # noqa: E402
from app.retrieval import hybrid, rerank  # noqa: E402

# Pairs chosen to exercise both retrieval styles.
DEFAULT_QUERIES = [
    ("doctor", "amoxicillin paediatric dosage"),        # exact drug name -> BM25 wins
    ("nurse", "how do I stop infections spreading"),     # conceptual -> dense wins
    ("technician", "calibration schedule frequency"),    # mixed
    ("billing_executive", "how do I submit a claim"),    # conceptual
]


def show(title: str, hits, limit: int = 5) -> None:
    print(f"\n  {title}")
    print("  " + "-" * 72)
    if not hits:
        print("    (nothing returned)")
        return
    for i, hit in enumerate(hits[:limit], 1):
        snippet = hit.text.replace("\n", " ")[:52]
        print(f"    {i}. [{hit.collection:9}] {hit.score:.4f}  {snippet}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default=None)
    ap.add_argument("--q", default=None, help="single query to test")
    opts = ap.parse_args()

    client = store.get_client()
    if not client.collection_exists(settings.qdrant_collection):
        print("No collection — run ingest.py first.")
        return 1

    queries = (
        [(opts.role or "admin", opts.q)] if opts.q else DEFAULT_QUERIES
    )

    for role, question in queries:
        print("\n" + "=" * 78)
        print(f"QUERY: \"{question}\"    (as {role})")
        print("=" * 78)

        dense = hybrid.dense_only_search(client, question, role)
        sparse = hybrid.sparse_only_search(client, question, role)
        fused = hybrid.hybrid_search(client, question, role)

        # Duplicate chunks crowd out real results and make reranker scores
        # tie. If this fires, re-run ingest.py --recreate.
        texts = [h.text for h in fused]
        if len(set(texts)) < len(texts):
            print(f"\n  !! WARNING: {len(texts) - len(set(texts))} duplicate chunk(s) "
                  "in the candidate set.")
            print("     Run: python backend/scripts/ingest.py --recreate")

        show("DENSE ONLY (semantic)", dense)
        show("BM25 ONLY (keyword)", sparse)
        show("HYBRID (RRF fused)", fused)

        # Which documents did each method find that the others missed?
        dense_docs = {h.text[:60] for h in dense[:3]}
        sparse_docs = {h.text[:60] for h in sparse[:3]}
        only_sparse = sparse_docs - dense_docs
        if only_sparse:
            print(f"\n  -> BM25 surfaced {len(only_sparse)} chunk(s) in its top 3 that")
            print("     dense search missed entirely. That is why hybrid exists.")

        print(f"\n  RERANKING {len(fused)} candidates -> top {settings.rerank_top_k}:")
        top = rerank.rerank(question, fused, log_scores=True)

        print(f"  Final context sent to the LLM ({len(top)} chunks):")
        for i, hit in enumerate(top, 1):
            print(f"    [{i}] {hit.source_document} > {hit.section_title[:44]}")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
