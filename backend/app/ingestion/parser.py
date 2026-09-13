"""Component 1: structure-aware parsing and hierarchical chunking.

The pipeline per document:

    PDF ──Docling──> DoclingDocument (a tree: headings, tables, paragraphs)
        ──HybridChunker──> chunks that respect that tree
        ──contextualize──> chunk text with its heading path prepended
        ──> Chunk objects carrying the full metadata schema

The collection a document belongs to is taken from its PARENT FOLDER name,
not from the filename. Folder names are something you control exactly;
filenames arrive however the dataset author felt that day.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from app.core import rbac
from app.core.config import settings

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One retrievable unit, with the exact metadata schema the brief requires."""

    text: str                  # what actually gets embedded (heading path + body)
    source_document: str       # original filename
    collection: str            # general | clinical | nursing | billing | equipment
    access_roles: List[str]    # derived from the RBAC matrix — never hand-written
    section_title: str         # heading this chunk sits under
    chunk_type: str            # text | table | heading | code
    extra: Dict[str, Any] = field(default_factory=dict)

    def content_id(self) -> str:
        """A deterministic ID derived from the chunk's content and origin.

        Using this as the Qdrant point ID makes ingestion IDEMPOTENT: running
        the script twice upserts the same IDs rather than inserting second
        copies. With a random uuid4 you silently accumulate duplicates, which
        shows up later as several retrieved chunks with identical text and
        identical reranker scores — crowding out genuinely different results.
        """
        seed = f"{self.source_document}|{self.section_title}|{self.text}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        # Format as a UUID string, which is what Qdrant accepts for point IDs.
        return str(uuid.UUID(digest[:32]))

    def payload(self) -> Dict[str, Any]:
        """The JSON stored alongside the vector in Qdrant.

        `access_roles` in here is what the query-time filter matches against.
        Get this field wrong and RBAC silently fails — which is why it is
        computed by rbac.roles_for() rather than typed by hand.
        """
        return {
            "text": self.text,
            "source_document": self.source_document,
            "collection": self.collection,
            "access_roles": self.access_roles,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            **self.extra,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def collection_of(path: Path, root: Optional[Path] = None) -> str:
    """Work out the collection from the folder the file sits in.

    data/documents/clinical/formulary.pdf  ->  "clinical"

    Falls back to "general" with a warning rather than crashing, so one
    stray file doesn't abort a 20-minute ingestion run.
    """
    root = root or settings.documents_dir
    try:
        relative = path.relative_to(root)
        folder = relative.parts[0].lower() if len(relative.parts) > 1 else ""
    except ValueError:
        folder = ""

    valid = {c.value for c in rbac.Collection}
    if folder in valid:
        return folder
    log.warning("'%s' is not in a recognised collection folder — filing as general", path.name)
    return rbac.Collection.GENERAL.value


def heading_path(chunk: Any) -> str:
    """Extract the heading trail Docling recorded for this chunk.

    Docling stores ancestor headings in chunk.meta.headings, e.g.
    ["Paediatric Dosing", "Amoxicillin"]. We join them into a breadcrumb so
    the embedded text knows where it came from.
    """
    try:
        headings = getattr(chunk.meta, "headings", None) or []
        return " > ".join(str(h).strip() for h in headings if str(h).strip())
    except AttributeError:
        return ""


def detect_chunk_type(chunk: Any, text: str) -> str:
    """Classify the chunk as text / table / heading / code.

    Docling records the original document item type. We read it where we can
    and fall back to shape heuristics — a line with several pipe characters
    is almost certainly a rendered table row.
    """
    try:
        items = getattr(chunk.meta, "doc_items", None) or []
        labels = {str(getattr(item, "label", "")).lower() for item in items}
        if any("table" in label for label in labels):
            return "table"
        if any("code" in label for label in labels):
            return "code"
        if any(label.endswith("header") or "title" in label for label in labels):
            return "heading"
    except AttributeError:
        pass

    if text.count("|") >= 4:
        return "table"
    stripped = text.strip()
    if stripped.startswith("```"):
        return "code"
    # A heading is short, single-line, and does NOT end in sentence
    # punctuation. Without that last condition a short paragraph like
    # "Check the valve seal before each use." gets misfiled as a heading.
    if (
        len(stripped) < 70
        and "\n" not in stripped
        and not stripped.endswith((".", "!", "?", ":", ";"))
    ):
        return "heading"
    return "text"


def contextualize(body: str, headings: str, source: str) -> str:
    """Prepend the heading breadcrumb to the text that gets embedded.

    This is the single highest-leverage line in the ingestion pipeline.
    "25mg twice daily" is unretrievable; the same text under
    "Amoxicillin > Paediatric Dosing" is both retrievable and safe to show.
    """
    parts = [p for p in (headings, body.strip()) if p]
    return "\n\n".join(parts) if parts else body.strip()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def build_chunker():
    """Create Docling's HybridChunker.

    'Hybrid' here means two passes, which is exactly what the brief asks for:
      1. split along document STRUCTURE (section -> subsection -> table)
      2. then merge/split to respect a TOKEN limit

    The tokenizer must be the same one the embedding model uses, or the token
    counts are meaningless and chunks silently overflow the model's window.
    """
    from docling.chunking import HybridChunker

    return HybridChunker(
        tokenizer=settings.dense_model,
        max_tokens=settings.max_tokens_per_chunk,
        merge_peers=True,  # glue tiny adjacent fragments back together
    )


def parse_document(path: Path, converter=None, chunker=None) -> List[Chunk]:
    """Parse one PDF or Markdown file into metadata-carrying chunks."""
    from docling.document_converter import DocumentConverter

    converter = converter or DocumentConverter()
    chunker = chunker or build_chunker()

    log.info("parsing %s", path.name)
    try:
        result = converter.convert(str(path))
        document = result.document
    except Exception as err:  # noqa: BLE001
        log.error("could not parse %s: %s", path.name, err)
        return []

    collection = collection_of(path)
    access_roles = rbac.roles_for(collection)  # derived, never hand-written

    chunks: List[Chunk] = []
    for raw in chunker.chunk(dl_doc=document):
        body = (raw.text or "").strip()
        if len(body) < 20:  # skip page numbers, stray artefacts
            continue

        headings = heading_path(raw)
        chunks.append(
            Chunk(
                text=contextualize(body, headings, path.name),
                source_document=path.name,
                collection=collection,
                access_roles=access_roles,
                section_title=headings or "(untitled section)",
                chunk_type=detect_chunk_type(raw, body),
            )
        )

    log.info("  %s -> %d chunks [%s]", path.name, len(chunks), collection)
    return chunks


def discover_documents(root: Optional[Path] = None) -> Iterator[Path]:
    """Find every PDF and Markdown file under the documents folder."""
    root = root or settings.documents_dir
    for pattern in ("*.pdf", "*.md", "*.markdown"):
        yield from sorted(root.rglob(pattern))


def parse_all(root: Optional[Path] = None) -> List[Chunk]:
    """Parse every document. Loads the converter once — it is expensive."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    chunker = build_chunker()

    chunks: List[Chunk] = []
    for path in discover_documents(root):
        chunks.extend(parse_document(path, converter, chunker))
    return chunks
