"""Run the full ingestion pipeline.

    python backend/scripts/ingest.py            # add to existing collection
    python backend/scripts/ingest.py --recreate # wipe and rebuild
    python backend/scripts/ingest.py --dry-run  # parse only, don't index

Run this ONCE as a standalone script before your demo. First execution
downloads Docling's layout models and the embedding models, which can take
several minutes. You do not want that happening live.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.ingestion import parser, store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    args = argparse.ArgumentParser(description="Ingest MediAssist documents into Qdrant")
    args.add_argument("--recreate", action="store_true", help="delete and rebuild the collection")
    args.add_argument("--dry-run", action="store_true", help="parse and report, do not index")
    opts = args.parse_args()

    print(f"\nReading from {settings.documents_dir}\n")

    files = list(parser.discover_documents())
    if not files:
        print("No PDFs or Markdown files found.")
        print("Expected layout:\n")
        print("  data/documents/general/...pdf")
        print("  data/documents/clinical/...pdf")
        print("  data/documents/nursing/...pdf")
        print("  data/documents/billing/...pdf")
        print("  data/documents/equipment/...pdf\n")
        return 1

    print(f"Found {len(files)} document(s):")
    for f in files:
        print(f"  [{parser.collection_of(f):9}] {f.name}")
    print()

    chunks = parser.parse_all()
    if not chunks:
        print("Parsing produced no chunks. Check the files are readable PDFs.")
        return 1

    # --- report what we built ------------------------------------------- #
    by_collection = Counter(c.collection for c in chunks)
    by_type = Counter(c.chunk_type for c in chunks)

    print(f"\n{'=' * 58}")
    print(f"{len(chunks)} chunks from {len(files)} documents")
    print(f"{'=' * 58}")
    print("\nBy collection:")
    for coll, n in sorted(by_collection.items()):
        roles = ", ".join(chunks[0].access_roles) if False else ""
        print(f"  {coll:10} {n:5}")
    print("\nBy chunk type:")
    for ctype, n in sorted(by_type.items()):
        print(f"  {ctype:10} {n:5}")

    # Show a real chunk so you can SEE the heading context in the text.
    sample = next((c for c in chunks if c.chunk_type == "table"), chunks[0])
    print(f"\n{'-' * 58}")
    print("Sample chunk (check the heading appears in the text):")
    print(f"{'-' * 58}")
    print(f"  source_document : {sample.source_document}")
    print(f"  collection      : {sample.collection}")
    print(f"  access_roles    : {sample.access_roles}")
    print(f"  section_title   : {sample.section_title}")
    print(f"  chunk_type      : {sample.chunk_type}")
    print(f"  text            : {sample.text[:400]}...")
    print(f"{'-' * 58}\n")

    if opts.dry_run:
        print("Dry run — nothing indexed.")
        return 0

    client = store.get_client()
    store.create_collection(client, recreate=opts.recreate)
    count = store.index_chunks(client, chunks)

    print(f"\nIndexed {count} chunks into '{settings.qdrant_collection}'.")
    print("\nStored per collection:")
    for coll, n in store.collection_stats(client).items():
        print(f"  {coll:10} {n:5}")
    print(f"\nInspect at {settings.qdrant_url}/dashboard\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
