from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError
from bson import ObjectId
from bson.errors import InvalidId

logger = logging.getLogger("autoassess")


# ============================================================
# Configuration
# ============================================================
#
# Everything is driven by env vars so nothing is hard-coded:
#
#   MONGODB_URI
#       connection string (default: local mongod)
#
#   MONGODB_DB
#       database name (default: "autoassess")
#
#   MONGODB_SCANNED_COLLECTION
#       collection for OCR'd (scanned) documents
#       (default: "scanned_documents")
#
#   MONGODB_MODEL_ANSWERS_COLLECTION
#       collection for documents that went through
#       plain markdown conversion, no OCR needed
#       (default: "model_answers")
#
#   LATEST_MODELS_LIMIT
#       Number of latest model answers returned by the API.
#       (default: 10)

LATEST_MODELS_LIMIT = 5

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017",
)

MONGODB_DB = os.getenv(
    "MONGODB_DB",
    "autoassess",
)

SCANNED_DOCUMENTS_COLLECTION = os.getenv(
    "MONGODB_SCANNED_COLLECTION",
    "scanned_documents",
)

MODEL_ANSWERS_COLLECTION = os.getenv(
    "MONGODB_MODEL_ANSWERS_COLLECTION",
    "model_answers",
)

# ============================================================
# Application limits
# ============================================================

LATEST_MODELS_LIMIT = int(
    os.getenv("LATEST_MODELS_LIMIT", "10")
)


_client: Optional[MongoClient] = None
_db: Optional[Database] = None


# ============================================================
# Lifecycle
# ============================================================

def connect() -> None:
    """Open the MongoDB connection. Call once on app startup."""

    global _client, _db

    logger.info("Connecting to MongoDB at %s", MONGODB_URI)

    _client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,
    )

    # Fail fast if MongoDB is unreachable.
    try:
        _client.admin.command("ping")

    except PyMongoError:
        logger.exception(
            "Could not reach MongoDB at %s",
            MONGODB_URI,
        )
        raise

    _db = _client[MONGODB_DB]

    # Helpful, non-unique indexes for common lookups.
    for collection_name in (
        SCANNED_DOCUMENTS_COLLECTION,
        MODEL_ANSWERS_COLLECTION,
    ):
        collection = _db[collection_name]

        collection.create_index("filename")
        collection.create_index("created_at")
        collection.create_index("type")

    logger.info(
        "MongoDB connected: db=%s scanned_collection=%s "
        "model_answers_collection=%s",
        MONGODB_DB,
        SCANNED_DOCUMENTS_COLLECTION,
        MODEL_ANSWERS_COLLECTION,
    )


def close() -> None:
    """Close the MongoDB connection. Call once on app shutdown."""

    global _client, _db

    if _client is not None:
        _client.close()

    _client = None
    _db = None


def get_collection(name: str) -> Collection:
    """Return a MongoDB collection by name."""

    if _db is None:
        raise RuntimeError(
            "MongoDB is not connected. "
            "Call db.connect() on startup first."
        )

    return _db[name]


# ============================================================
# Writes
# ============================================================

def save_document(
    payload: dict[str, Any],
    collection_name: str,
) -> Optional[str]:
    """
    Insert one processed-document record into the given collection.

    Returns the inserted document's string id, or None if the write
    failed.
    """

    record = dict(payload)

    record.setdefault(
        "created_at",
        datetime.now(timezone.utc),
    )

    try:
        collection = get_collection(collection_name)

        result = collection.insert_one(record)

        return str(result.inserted_id)

    except PyMongoError:
        logger.exception(
            "Failed to save document to MongoDB collection '%s'.",
            collection_name,
        )

        return None


# ============================================================
# Reads
# ============================================================

def get_document(
    document_id: str,
    collection_name: str,
) -> Optional[dict[str, Any]]:
    """
    Fetch one document by its string id.

    Returns None if the id is malformed, the document doesn't
    exist, or MongoDB is unreachable.
    """

    try:
        oid = ObjectId(document_id)

    except (InvalidId, TypeError, ValueError):
        logger.warning(
            "Invalid document_id '%s' for collection '%s'.",
            document_id,
            collection_name,
        )

        return None

    try:
        collection = get_collection(collection_name)

        doc = collection.find_one(
            {"_id": oid}
        )

    except PyMongoError:
        logger.exception(
            "Failed to fetch document '%s' from MongoDB collection '%s'.",
            document_id,
            collection_name,
        )

        return None

    if doc is not None:
        doc["_id"] = str(doc["_id"])

    return doc


def get_latest_document(
    collection_name: str,
    filters: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """
    Fetch the most recently created document in a collection.

    Returns None if nothing matches or MongoDB is unreachable.
    """

    try:
        collection = get_collection(collection_name)

        doc = collection.find_one(
            filters or {},
            sort=[("created_at", -1)],
        )

    except PyMongoError:
        logger.exception(
            "Failed to fetch latest document from MongoDB collection '%s'.",
            collection_name,
        )

        return None

    if doc is not None:
        doc["_id"] = str(doc["_id"])

    return doc


def get_documents(
    collection_name: str,
    filters: Optional[dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Fetch documents from a collection.

    Documents are returned in descending order by created_at,
    meaning the latest inserted documents are returned first.

    Args:
        collection_name:
            MongoDB collection name.

        filters:
            Optional MongoDB filter.

        limit:
            Maximum number of documents to return.
            If None, all matching documents are returned.

    Returns:
        List of documents. Returns an empty list if nothing
        matches or MongoDB is unreachable.
    """

    try:
        collection = get_collection(collection_name)

        cursor = collection.find(
            filters or {},
            {
                "_id": 1,
                "filename": 1,
                "created_at": 1,
            },
        ).sort(
            "created_at",
            -1,
        )

        if limit is not None:
            cursor = cursor.limit(limit)

        documents = list(cursor)

    except PyMongoError:
        logger.exception(
            "Failed to fetch documents from MongoDB collection '%s'.",
            collection_name,
        )

        return []

    for document in documents:
        document["_id"] = str(document["_id"])

    return documents