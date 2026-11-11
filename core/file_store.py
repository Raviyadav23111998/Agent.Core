"""File store — a Python port of the SMSystems `FileService` file subsystem.

We deliberately mirror SMSystems (`SMSystems.Application/Staff/Services/
FileService.cs` + `system.files`) so attachment storage works "the same way"
across the org:

  * base path chosen by file type from config
      IMG → media.image_base_path   (SMSystems ImageBasePath)
      DOC → media.doc_base_path     (SMSystems DOCBasePath)
      AUD → media.audio_base_path   (our audio extension)
      VID → media.video_base_path   (our video extension)
      else → media.base_path        (SMSystems BasePath)
  * layout: {base}/{tenant}/{associated_to}/{associated_id}/{guid}{ext}
  * GUID filenames; original name kept in the DB row (ag_system.files)
  * images get a "{guid}_thumbnail500{ext}" variant alongside the original
  * the DB stores the RELATIVE file_path "{associated_to}/{associated_id}/{guid}{ext}";
    reads recombine {base}/{tenant}/ + file_path
  * base64 in / base64 out (also exactly what the Anthropic image/document
    content blocks need — see client/anthropic_client._convert_history)

This module owns ONLY the filesystem side (bytes in/out). The DB `ag_system.files`
row is written by the API repository/service, mirroring how SMSystems' FileService
commits the file write and the repo row in one UnitOfWork transaction — here the
API service orchestrates store.write() + repo.add_file() together.

It is also the S3-swap seam: replace FilesystemFileStore with an S3 driver and no
caller changes (same write/read_b64/delete signatures).
"""

from __future__ import annotations

import io
import os
import uuid
from typing import Optional

from core.config import app_config

# File-type codes — SMSystems uses IMG/DOC; we add AUD/VID for the audio/video
# arms of both attachment directions.
IMG, DOC, AUD, VID = "IMG", "DOC", "AUD", "VID"

# Map a MIME type to a SMSystems-style file-type code.
_MIME_KIND = {
    "image/": IMG,
    "application/pdf": DOC,
    "audio/": AUD,
    "video/": VID,
}


def kind_for_mime(media_type: str) -> str:
    mt = (media_type or "").lower()
    for prefix, code in _MIME_KIND.items():
        if mt.startswith(prefix) or mt == prefix:
            return code
    return DOC  # text/* and unknowns land under the generic DOC base path


class FilesystemFileStore:
    """Filesystem driver. Same call sites will accept an S3 driver later."""

    def _base_path(self, file_type: str) -> str:
        if file_type == IMG:
            base = app_config.get("media.image_base_path")
        elif file_type == DOC:
            base = app_config.get("media.doc_base_path")
        elif file_type == AUD:
            base = app_config.get("media.audio_base_path")
        elif file_type == VID:
            base = app_config.get("media.video_base_path")
        else:
            base = app_config.get("media.base_path")
        return base or "/var/agent-media/"

    def _tenant_root(self, file_type: str, tenant_id: str) -> str:
        # {base}/{tenant}/   (SMSystems: BasePath + TenantId + "//")
        return os.path.join(self._base_path(file_type), str(tenant_id or "default"))

    # Derived image variants: a small UI thumbnail and a bounded "vision" copy
    # sent to the model (keeps base64 under provider per-image limits).
    def _thumb_px(self) -> int:
        return int(app_config.get("media.thumbnail_px", "500") or 500)

    def _vision_px(self) -> int:
        return int(app_config.get("media.vision_px", "1568") or 1568)

    @staticmethod
    def _variant_rel(file_path: str, tag: str, px: int) -> str:
        dot = file_path.rfind(".")
        return (f"{file_path[:dot]}_{tag}{px}{file_path[dot:]}"
                if dot >= 0 else f"{file_path}_{tag}{px}")

    def write(self, *, file_type: str, extension: str, data: bytes,
              associated_to: str, associated_id, tenant_id: str,
              thumbnail: bool = False, thumbnail_px: int = None) -> dict:
        """Write bytes and (for images) a UI thumbnail + a bounded vision variant.
        Returns the DB-facing row fields: {guid, file_path, size_bytes}. `file_path`
        is RELATIVE, exactly like SMSystems (`{assoc_to}/{assoc_id}/{guid}{ext}`)."""
        ext = extension if extension.startswith(".") else f".{extension}" if extension else ""
        guid = uuid.uuid4().hex
        middle = os.path.join(str(associated_to), str(associated_id))
        abs_dir = os.path.join(self._tenant_root(file_type, tenant_id), middle)
        os.makedirs(abs_dir, exist_ok=True)

        base_name = f"{guid}{ext}"
        with open(os.path.join(abs_dir, base_name), "wb") as f:
            f.write(data)

        if file_type == IMG:
            for tag, px in (("thumbnail", self._thumb_px()), ("vision", self._vision_px())):
                variant = self._downscale(data, px)
                if variant is not None:
                    with open(os.path.join(abs_dir, f"{guid}_{tag}{px}{ext}"), "wb") as f:
                        f.write(variant)

        rel = f"{associated_to}/{associated_id}/{base_name}"
        return {"guid": guid, "file_path": rel, "size_bytes": len(data)}

    def read(self, *, file_type: str, tenant_id: str, file_path: str,
             variant: str = None) -> Optional[bytes]:
        """Read bytes. `file_path` is the relative DB path. `variant` ∈
        {None,'thumbnail','vision'} (images only). If a 'vision' variant is
        missing (legacy file, or media.vision_px changed since upload), the
        bounded image is DERIVED on the fly — the model is never handed an
        un-bounded original (which could exceed provider per-image limits)."""
        root = self._tenant_root(file_type, tenant_id)
        if file_type == IMG and variant in ("thumbnail", "vision"):
            px = self._thumb_px() if variant == "thumbnail" else self._vision_px()
            vpath = os.path.join(root, self._variant_rel(file_path, variant, px))
            if os.path.isfile(vpath):
                return self._read_raw(vpath)
            orig = self._read_raw(os.path.join(root, file_path))
            if orig is None:
                return None
            if variant == "vision":
                derived = self._downscale(orig, px)
                return derived if derived is not None else orig
            return orig  # thumbnail missing → original is fine for display
        return self._read_raw(os.path.join(root, file_path))

    @staticmethod
    def _read_raw(abs_path: str) -> Optional[bytes]:
        if not os.path.isfile(abs_path):
            return None
        with open(abs_path, "rb") as f:
            return f.read()

    def read_b64(self, **kwargs) -> Optional[str]:
        import base64
        data = self.read(**kwargs)
        return base64.b64encode(data).decode("ascii") if data is not None else None

    def delete(self, *, file_type: str, tenant_id: str, file_path: str) -> None:
        root = self._tenant_root(file_type, tenant_id)
        paths = [os.path.join(root, file_path)]
        if file_type == IMG:
            for tag, px in (("thumbnail", self._thumb_px()), ("vision", self._vision_px())):
                paths.append(os.path.join(root, self._variant_rel(file_path, tag, px)))
        for p in paths:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass

    @staticmethod
    def _downscale(data: bytes, px: int) -> Optional[bytes]:
        """Return a copy whose long edge is ≤ px. Returns None on non-images.
        Always re-encodes so the variant file exists even when already smaller
        (callers rely on the variant being present)."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            fmt = img.format or "PNG"
            img.thumbnail((px, px))  # only shrinks; no-op if already smaller
            out = io.BytesIO()
            if fmt in ("JPEG", "JPG"):
                img.convert("RGB").save(out, format="JPEG", quality=85)
            else:
                img.save(out, format=fmt)
            data2 = out.getvalue()
            # Safety net: a dense PNG can still exceed provider per-image limits
            # even at the pixel cap — fall back to JPEG so base64 stays well under.
            if len(data2) > 4_500_000 and fmt not in ("JPEG", "JPG"):
                out = io.BytesIO()
                img.convert("RGB").save(out, format="JPEG", quality=82)
                data2 = out.getvalue()
            return data2
        except Exception:
            return None  # best-effort; the original is always written


# Process-wide default store (swap the class here to move to S3).
file_store = FilesystemFileStore()


def downscale_image(data: bytes, px: int = None) -> bytes:
    """Bound an image's long edge (default media.vision_px) for sending to a
    model — used by channel ingress (Telegram/Teams photos) that bypass the
    upload path. Returns the original on non-images / failure."""
    if px is None:
        px = int(app_config.get("media.vision_px", "1568") or 1568)
    out = FilesystemFileStore._downscale(data, px)
    return out if out is not None else data
