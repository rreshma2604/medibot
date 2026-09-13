"""Start the API without worrying about paths or PATH.

    python run_api.py

Run from the PROJECT ROOT (the folder with backend/ and data/ in it).
This does what `uvicorn app.api.main:app` does, but it adds backend/ to
sys.path itself and reports import errors properly instead of hiding them
behind "Could not import module".
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

if not BACKEND.exists():
    print(f"Can't find {BACKEND}")
    print("Run this from the project root — the folder containing backend/ and data/.")
    raise SystemExit(1)

sys.path.insert(0, str(BACKEND))

# Import here, outside uvicorn, so a real traceback reaches you.
try:
    from app.api.main import app  # noqa: F401
except ImportError as err:
    print("\nCouldn't import the API. The real error:\n")
    print(f"  {type(err).__name__}: {err}\n")
    print("Common causes:")
    print("  - a file saved with a capital letter (Main.py instead of main.py)")
    print("  - a missing __init__.py in one of the app/ subfolders")
    print("  - a package not installed: pip install -r backend/requirements.txt\n")
    raise SystemExit(1)

if __name__ == "__main__":
    import uvicorn

    print("\n  API      http://localhost:8000")
    print("  Docs     http://localhost:8000/docs")
    print("  Health   http://localhost:8000/health\n")
    print("  Stop this before running ingest.py — embedded Qdrant allows one process.\n")

    uvicorn.run(
        "app.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(BACKEND)],
    )
