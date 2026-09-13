"""Role-based access control.

This module is the ONE place that decides who can see what. Nothing else in
the codebase should hardcode a role or a collection name — if you find
yourself typing "nurse" anywhere else, import from here instead.

Why this matters: RBAC bugs are almost always caused by the rules living in
two places that drift apart. One source of truth removes that whole class of
bug, and it means the ingestion script and the retrieval query cannot
disagree about who may read a document.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Set


class Role(str, Enum):
    """The five staff roles.

    Inheriting from `str` as well as `Enum` is a small trick: it means
    Role.NURSE == "nurse" is True, so these work seamlessly with JSON,
    FastAPI request bodies and Qdrant filters without conversion.
    """

    DOCTOR = "doctor"
    NURSE = "nurse"
    BILLING_EXECUTIVE = "billing_executive"
    TECHNICIAN = "technician"
    ADMIN = "admin"


class Collection(str, Enum):
    """The five document collections from the assignment brief."""

    GENERAL = "general"
    CLINICAL = "clinical"
    NURSING = "nursing"
    BILLING = "billing"
    EQUIPMENT = "equipment"


# The access matrix, transcribed directly from the brief.
# Read this as: "a user with THIS role may read chunks from THESE collections".
ROLE_COLLECTIONS: Dict[Role, Set[Collection]] = {
    Role.DOCTOR: {Collection.GENERAL, Collection.CLINICAL, Collection.NURSING},
    Role.NURSE: {Collection.GENERAL, Collection.NURSING},
    Role.BILLING_EXECUTIVE: {Collection.GENERAL, Collection.BILLING},
    Role.TECHNICIAN: {Collection.GENERAL, Collection.EQUIPMENT},
    Role.ADMIN: set(Collection),  # admin sees everything
}

# Only these roles may run SQL RAG against mediassist.db.
SQL_ROLES: Set[Role] = {Role.BILLING_EXECUTIVE, Role.ADMIN}


def collections_for(role: Role) -> List[str]:
    """Which collections may this role read? Returns plain strings, sorted.

    Sorted so the output is deterministic — helpful when you're comparing
    API responses in tests or screenshots.
    """
    return sorted(c.value for c in ROLE_COLLECTIONS.get(Role(role), set()))


def roles_for(collection: Collection) -> List[str]:
    """The inverse lookup: which roles may read this collection?

    This is what the ingestion script writes into each chunk's
    `access_roles` metadata field. Deriving it from the same matrix means
    the stored metadata can never contradict the query-time filter.
    """
    collection = Collection(collection)
    return sorted(
        role.value
        for role, allowed in ROLE_COLLECTIONS.items()
        if collection in allowed
    )


def can_access(role: Role, collection: Collection) -> bool:
    """Would this role be allowed to see this collection?

    Used for the friendly refusal message in the frontend. Note it is NOT
    used to filter retrieval results — that happens inside the Qdrant query
    itself. This function is for explaining a refusal, never for enforcing it.
    """
    return Collection(collection) in ROLE_COLLECTIONS.get(Role(role), set())


def can_use_sql(role: Role) -> bool:
    """May this role query the relational database?"""
    return Role(role) in SQL_ROLES


def refusal_message(role: Role, blocked: Collection | None = None) -> str:
    """The user-facing 'you can't see that' message.

    The brief explicitly asks for something informative rather than a generic
    error, so we name the role, name what they CAN see, and avoid confirming
    any detail about the restricted content.
    """
    allowed = collections_for(role)
    readable = ", ".join(allowed[:-1]) + f" and {allowed[-1]}" if len(allowed) > 1 else allowed[0]
    who = Role(role).value.replace("_", " ")

    if blocked:
        return (
            f"As a {who}, you don't have access to {Collection(blocked).value} "
            f"documents. I can only answer questions from the {readable} collections."
        )
    return (
        f"I couldn't find an answer in the documents available to you. "
        f"As a {who}, I can only search the {readable} collections."
    )


# Demo accounts for the login endpoint.
# Plain-text passwords are fine ONLY because this is a graded demo with
# fictional users. In anything real these would be hashed with bcrypt or
# argon2 and stored in a database, never in source control.
DEMO_USERS: Dict[str, Dict[str, str]] = {
    "dr.mehta": {"password": "doctor123", "role": Role.DOCTOR.value, "name": "Dr. Mehta"},
    "nurse.priya": {"password": "nurse123", "role": Role.NURSE.value, "name": "Nurse Priya"},
    "billing.ravi": {"password": "billing123", "role": Role.BILLING_EXECUTIVE.value, "name": "Ravi (Billing)"},
    "tech.anand": {"password": "tech123", "role": Role.TECHNICIAN.value, "name": "Anand (Equipment)"},
    "admin.sys": {"password": "admin123", "role": Role.ADMIN.value, "name": "System Admin"},
}
