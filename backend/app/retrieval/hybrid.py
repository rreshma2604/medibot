"""Component 2: Hybrid retrieval (dense + BM25) with RBAC enforced in-query.

The shape of a hybrid query in Qdrant:

    query_points(
        prefetch=[ dense branch, sparse branch ],   <- two searches, one request
        query=FusionQuery(RRF),                     <- fused server-side
        query_filter=<RBAC>,                        <- applied to everything
    )

Both branches carry the RBAC filter, and so does the outer query. That is
belt and braces on purpose: a filter on a prefetch branch restricts what that
branch retrieves, and the outer filter restricts the fused result. Either
alone would be sufficient; both together means a future edit to one cannot
silently open a hole.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.ingestion.store import DENSE, SPARSE, rbac_filter

log = logging.getLogger(__name__)


@dataclass
class Hit:
    """One retrieved chunk, with whatever scores it has picked up so far."""

    text: str
    source_document: str
    section_title: str
    collection: str
    chunk_type: str
    score: float                      # fusion score from hybrid search
    rerank_score: Optional[float] = None   # filled in by the reranker

    @property
    def payload(self) -> Dict[str, Any]:
        """Kept so calling code can treat a Hit like a raw Qdrant point."""
        return {
            "text": self.text,
            "source_document": self.source_document,
            "section_title": self.section_title,
            "collection": self.collection,
            "chunk_type": self.chunk_type,
        }

    def citation(self) -> Dict[str, str]:
        """The {source_document, section_title, collection} the brief requires."""
        return {
            "source_document": self.source_document,
            "section_title": self.section_title,
            "collection": self.collection,
        }


@lru_cache(maxsize=1)
def _embedders():
    """Load embedding models once per process.

    lru_cache is doing real work here — loading these on every query would
    add seconds to each request. Cached, the models stay resident.
    """
    from fastembed import SparseTextEmbedding, TextEmbedding

    return (
        TextEmbedding(model_name=settings.dense_model),
        SparseTextEmbedding(model_name=settings.sparse_model),
    )


def embed_query(question: str):
    """Turn the question into both a dense and a sparse vector."""
    dense_model, sparse_model = _embedders()
    dense_vec = next(iter(dense_model.embed([question]))).tolist()
    raw_sparse = next(iter(sparse_model.embed([question])))
    sparse_vec = models.SparseVector(
        indices=raw_sparse.indices.tolist(),
        values=raw_sparse.values.tolist(),
    )
    return dense_vec, sparse_vec


def _to_hits(points) -> List[Hit]:
    hits = []
    for point in points:
        p = point.payload or {}
        hits.append(
            Hit(
                text=p.get("text", ""),
                source_document=p.get("source_document", "unknown"),
                section_title=p.get("section_title", ""),
                collection=p.get("collection", ""),
                chunk_type=p.get("chunk_type", "text"),
                score=float(point.score),
            )
        )
    return hits


def hybrid_search(
    client: QdrantClient,
    question: str,
    role: str,
    limit: Optional[int] = None,
    collection: Optional[str] = None,
) -> List[Hit]:
    """Dense + BM25 in one request, fused with RRF, filtered by role.

    Returns the broad candidate set (default 10) for the reranker to narrow.
    """
    limit = limit or settings.retrieve_top_k
    dense_vec, sparse_vec = embed_query(question)
    access = rbac_filter(role, collection)

    # Over-fetch in each branch so fusion has material to work with.
    branch_limit = limit * 2

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using=DENSE,
                limit=branch_limit,
                filter=access,
            ),
            models.Prefetch(
                query=sparse_vec,
                using=SPARSE,
                limit=branch_limit,
                filter=access,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=access,
        limit=limit,
        with_payload=True,
    )
    return _to_hits(response.points)


def dense_only_search(
    client: QdrantClient,
    question: str,
    role: str,
    limit: Optional[int] = None,
) -> List[Hit]:
    """Dense search alone — used ONLY to demonstrate that hybrid is better.

    The brief asks for retrieval quality "demonstrably better than
    dense-only", which means you need the dense-only baseline to compare
    against. Not used in the live /chat path.
    """
    limit = limit or settings.retrieve_top_k
    dense_vec, _ = embed_query(question)

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=dense_vec,
        using=DENSE,
        query_filter=rbac_filter(role),
        limit=limit,
        with_payload=True,
    )
    return _to_hits(response.points)


def sparse_only_search(
    client: QdrantClient,
    question: str,
    role: str,
    limit: Optional[int] = None,
) -> List[Hit]:
    """BM25 alone — the other half of the comparison."""
    limit = limit or settings.retrieve_top_k
    _, sparse_vec = embed_query(question)

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=sparse_vec,
        using=SPARSE,
        query_filter=rbac_filter(role),
        limit=limit,
        with_payload=True,
    )
    return _to_hits(response.points)
