"""Component 5: the FastAPI backend.

    uvicorn app.api.main:app --reload --port 8000   (run from backend/)

Then open http://localhost:8000/docs for an interactive UI you can test the
whole system from, before any frontend exists.

Note on embedded Qdrant: only one process may hold the storage folder, so
stop this server before running ingest.py.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from app.core import rbac
from app.core.config import settings
from app.ingestion import store
from app.retrieval import generate, hybrid, rerank, sql_rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open Qdrant once at startup, not per request."""
    log.info("connecting to Qdrant (%s)", settings.qdrant_mode)
    state["qdrant"] = store.get_client()
    yield
    client = state.get("qdrant")
    if client:
        client.close()


app = FastAPI(
    title="MediBot API",
    description="Advanced RAG with role-based access control for MediAssist Health Network",
    version="1.0.0",
    lifespan=lifespan,
)

# The Next.js dev server runs on a different port, so it needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str
    collections: List[str]
    can_use_sql: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieval_type: Literal["hybrid_rag", "sql_rag", "blocked"]
    role: str
    blocked: bool = False


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

def make_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.token_expiry_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def current_role(authorization: Optional[str] = Header(None)) -> str:
    """Extract and verify the role from the bearer token.

    Critical: the role comes from the SIGNED TOKEN, never from the request
    body. If the client could post its own role, the entire RBAC model would
    be decorative — anyone could claim to be admin.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token. Call /login first.")

    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise HTTPException(401, "Invalid or expired token.")

    role = payload.get("role")
    try:
        return rbac.Role(role).value
    except ValueError:
        raise HTTPException(401, "Token carries an unknown role.")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> Dict[str, Any]:
    client = state.get("qdrant")
    indexed = False
    counts: Dict[str, int] = {}
    if client:
        try:
            indexed = client.collection_exists(settings.qdrant_collection)
            if indexed:
                counts = store.collection_stats(client)
        except Exception:  # noqa: BLE001
            pass
    return {
        "status": "ok",
        "qdrant_mode": settings.qdrant_mode,
        "index_ready": indexed,
        "chunks_per_collection": counts,
        "database_present": settings.sqlite_path.exists(),
    }


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    user = rbac.DEMO_USERS.get(request.username)
    if not user or user["password"] != request.password:
        # Same message either way — don't reveal which usernames exist.
        raise HTTPException(401, "Incorrect username or password.")

    role = user["role"]
    return LoginResponse(
        access_token=make_token(request.username, role),
        role=role,
        name=user["name"],
        collections=rbac.collections_for(role),
        can_use_sql=rbac.can_use_sql(role),
    )


@app.get("/collections/{role}")
def collections(role: str) -> Dict[str, Any]:
    try:
        validated = rbac.Role(role)
    except ValueError:
        raise HTTPException(404, f"Unknown role '{role}'.")
    return {
        "role": validated.value,
        "collections": rbac.collections_for(validated),
        "can_use_sql": rbac.can_use_sql(validated),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, role: str = Depends(current_role)) -> ChatResponse:
    """The main RAG endpoint.

    Routing:
      analytical question + permitted role -> SQL RAG
      analytical question + other role      -> friendly refusal
      everything else                       -> hybrid retrieval + rerank + LLM
    """
    question = request.question.strip()

    # --- analytical branch ---------------------------------------------- #
    if generate.is_analytical(question):
        if rbac.can_use_sql(role):
            answer = sql_rag.sql_rag_chain(question)
            return ChatResponse(
                answer=answer,
                sources=[Source(
                    source_document="mediassist.db",
                    section_title="claims / maintenance_tickets",
                    collection="database",
                )],
                retrieval_type="sql_rag",
                role=role,
            )
        # Not permitted: say so, and fall through to documents anyway, since
        # the question may well be answerable from a policy PDF.
        log.info("%s attempted an analytical query — SQL not permitted", role)

    # --- document branch ------------------------------------------------- #
    client = state.get("qdrant")
    if client is None or not client.collection_exists(settings.qdrant_collection):
        raise HTTPException(503, "Index not ready. Run ingest.py first.")

    candidates = hybrid.hybrid_search(client, question, role=role)

    if not candidates:
        # Nothing the role may see matched. This is the RBAC refusal path.
        return ChatResponse(
            answer=rbac.refusal_message(role),
            sources=[],
            retrieval_type="blocked",
            role=role,
            blocked=True,
        )

    top = rerank.rerank(question, candidates)
    ok, answer = generate.answer_from_context(question, top)
    if not ok:
        raise HTTPException(502, answer)

    return ChatResponse(
        answer=answer,
        sources=[Source(**hit.citation()) for hit in top],
        retrieval_type="hybrid_rag",
        role=role,
    )
