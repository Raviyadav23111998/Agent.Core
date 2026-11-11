"""Core-side insert for the canonical ag_system.files row.

Part 1 persists file rows from the API (FileRepository). Part 2 generates media
INSIDE the agent loop (core), so the generate_* tools need to write the same
canonical row without importing the API layer. Same columns as
api/repositories/file_repo.py — keep them in sync.
"""
from typing import Optional

_COLS = ("file_id, file_type, associated_to, associated_id, file_path, "
         "file_name, extension, size_bytes, mime_type, created_by, created_at")


_KIND = {"IMG": "image", "DOC": "document", "AUD": "audio", "VID": "video"}


def insert_file(db, *, file_type: str, associated_to: str, associated_id: int,
                file_path: str, file_name: str = None, extension: str = None,
                mime_type: str = None, size_bytes: int = 0,
                created_by: Optional[int] = None) -> int:
    row = db.execute_returning(f"""
        INSERT INTO ag_system.files
            (file_type, associated_to, associated_id, file_path,
             file_name, extension, mime_type, size_bytes, created_by)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING {_COLS}
    """, (file_type, associated_to, associated_id, file_path,
          file_name, extension, mime_type, size_bytes, created_by))
    return row["file_id"]


def load_file(db, file_id: int):
    """Load a stored file's bytes + metadata by id (for channel egress —
    Telegram/Teams sending a generated image). Returns
    {bytes, mime_type, kind, filename} or None."""
    row = db.fetch_one(
        "SELECT file_type, file_path, file_name, mime_type "
        "FROM ag_system.files WHERE file_id = %s", (file_id,))
    if not row:
        return None
    from core.file_store import file_store
    from core.tenant_context import current_tenant
    data = file_store.read(file_type=row["file_type"],
                           tenant_id=current_tenant().tenant_id,
                           file_path=row["file_path"])
    if data is None:
        return None
    return {"bytes": data, "mime_type": row.get("mime_type"),
            "kind": _KIND.get(row["file_type"], "document"),
            "filename": row.get("file_name") or "file"}
