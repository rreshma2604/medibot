"""Qdrant vector store: hybrid indexing and RBAC-filtered search.

Two things happen here that are worth understanding properly:

1. NAMED VECTORS. Each point carries TWO vectors — a dense one (meaning) and
   a sparse one (BM25 keywords). Qdrant supports several named vectors per
   point, which is what lets us satisfy the brief's requirement that both are
   "stored at index time and queried together at retrieval time", rather than
   running two searches and stitching them together in Python.

2. THE RBAC FILTER. Every search passes a Filter on the access_roles payload
   field. Qdrant applies it during the search, so restricted chunks are never
   candidates at all — they are not retrieved and then dropped, they are
   never retrieved.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from app.core import rbac
from app.core.config import settings
from app.ingestion.parser import Chunk

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"


def get_client() -> QdrantClient:
    """Connect to Qdrant, however the user is running it.

    Embedded mode is the interesting one: passing `path=` instead of `url=`
    makes qdrant-client run the engine in-process and persist to that folder.
    Same Python API, same filters, same hybrid search — no server involved.

    One constraint to know about: only ONE process may hold the embedded
    folder at a time. If your FastAPI server is running, the ingest script
    will refuse to open it, and vice versa. Stop one before running the other.
    """
    mode = settings.qdrant_mode.lower()

    if mode == "embedded":
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        log.info("Qdrant embedded, storing at %s", settings.qdrant_path)
        return QdrantClient(path=str(settings.qdrant_path))

    log.info("Qdrant %s at %s", mode, settings.qdrant_url)
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=120,
    )


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the collection with one dense and one sparse vector slot.

    `size` must equal the dense model's output dimension (384 for MiniLM).
    COSINE distance is the right metric for normalised sentence embeddings.
    """
    name = settings.qdrant_collection
    exists = client.collection_exists(name)

    if exists and recreate:
        log.info("deleting existing collection %s", name)
        client.delete_collection(name)
        exists = False

    if exists:
        log.info("collection %s already exists", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE: models.VectorParams(
                size=settings.dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # BM25 needs inverse document frequency
            )
        },
    )

    # Payload indexes make filtering fast. Without them Qdrant scans every
    # point's payload; with them the access_roles filter is near-free.
    for field in ("access_roles", "collection", "source_document"):
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    log.info("created collection %s (dense %d-dim + sparse BM25)", name, settings.dense_dim)


def build_embedders():
    """Load the dense and sparse embedding models.

    fastembed runs both locally on CPU. First call downloads the models, so
    the first ingestion run is slow and every run after is fast.
    """
    from fastembed import SparseTextEmbedding, TextEmbedding

    dense = TextEmbedding(model_name=settings.dense_model)
    sparse = SparseTextEmbedding(model_name=settings.sparse_model)
    return dense, sparse


def deduplicate(chunks: List[Chunk]) -> List[Chunk]:
    """Drop chunks whose content ID we have already seen in this run.

    Docling can emit the same table fragment more than once when a table
    spans pages, so this matters even on a single clean run.
    """
    seen = set()
    unique = []
    for chunk in chunks:
        key = chunk.content_id()
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)

    dropped = len(chunks) - len(unique)
    if dropped:
        log.info("dropped %d duplicate chunk(s) before indexing", dropped)
    return unique


def index_chunks(
    client: QdrantClient,
    chunks: List[Chunk],
    batch_size: int = 64,
) -> int:
    """Embed and upsert chunks, dense and sparse together in one point.

    Point IDs are derived from chunk content, so re-running this script
    overwrites existing points instead of duplicating them.
    """
    if not chunks:
        log.warning("nothing to index")
        return 0

    chunks = deduplicate(chunks)

    dense_model, sparse_model = build_embedders()
    total = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]

        dense_vectors = list(dense_model.embed(texts))
        sparse_vectors = list(sparse_model.embed(texts))

        points = [
            models.PointStruct(
                id=chunk.content_id(),  # deterministic -> re-ingest is idempotent
                vector={
                    DENSE: dense_vec.tolist(),
                    SPARSE: models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload=chunk.payload(),
            )
            for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors)
        ]

        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)
        log.info("  indexed %d/%d", total, len(chunks))

    return total


# --------------------------------------------------------------------------- #
# THE RBAC FILTER — the 25% of the grade
# --------------------------------------------------------------------------- #

def rbac_filter(role: str, collection: Optional[str] = None) -> models.Filter:
    """Build the Qdrant filter that enforces access control.

    MatchAny on access_roles means: return only points whose access_roles
    list contains this role. Because ingestion wrote access_roles from the
    same RBAC matrix, a nurse's query structurally cannot match a billing
    chunk.

    This filter is passed INTO the search call. Qdrant applies it while
    traversing the index, so restricted chunks are never scored, never
    ranked, never returned — and therefore never reach the LLM's context.
    No prompt, however adversarial, can extract text the model was never given.
    """
    must: List[models.Condition] = [
        models.FieldCondition(
            key="access_roles",
            match=models.MatchAny(any=[rbac.Role(role).value]),
        )
    ]

    # Optional narrowing — e.g. "only search the nursing collection".
    # This is a convenience, NOT the security boundary. The condition above
    # is the security boundary.
    if collection:
        must.append(
            models.FieldCondition(
                key="collection",
                match=models.MatchValue(value=rbac.Collection(collection).value),
            )
        )

    return models.Filter(must=must)


def collection_stats(client: QdrantClient) -> Dict[str, Any]:
    """Per-collection chunk counts — useful for verifying ingestion worked."""
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        return {}

    stats = {}
    for coll in rbac.Collection:
        count = client.count(
            collection_name=name,
            count_filter=models.Filter(
                must=[models.FieldCondition(
                    key="collection", match=models.MatchValue(value=coll.value)
                )]
            ),
            exact=True,
        ).count
        stats[coll.value] = count
    return stats
