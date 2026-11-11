import logging
import threading
from cryptography.fernet import Fernet
import base64
import os
import hashlib

logger = logging.getLogger(__name__)

_fernet_instance = None
_using_fallback_key = False
_fernet_init_lock = threading.Lock()


def _get_fernet():
    global _fernet_instance, _using_fallback_key
    # Fast path: already initialized, no lock needed for reads
    if _fernet_instance is not None:
        return _fernet_instance

    # Slow path: serialize initialization to prevent double-init race
    with _fernet_init_lock:
        # Double-check: another thread may have initialized while we waited
        if _fernet_instance is not None:
            return _fernet_instance

        from core.config import app_config

        key_str = app_config.get_global("secret.encryption_key")

        if not key_str:
            env = (os.environ.get("AG_ENVIRONMENT", "")
                   or os.environ.get("AG_ENV", "")).strip().lower()
            if env in ("prod", "production"):
                raise RuntimeError(
                    "secret.encryption_key is REQUIRED in production. "
                    "Set it in ag_system.configs (a urlsafe-base64 32-byte Fernet key) "
                    "before starting. Refusing to derive a key from the JWT secret."
                )
            _using_fallback_key = True
            logger.warning(
                "\u26a0 SECURITY: No secret.encryption_key found. "
                "Deriving key from JWT private key hash. "
                "Set 'secret.encryption_key' in ag_system.configs for production! "
                "If the JWT key rotates, ALL vault secrets become undecryptable."
            )
            base_material = app_config.get_global("secret.private_key", "default-insecure-seed").encode('utf-8')
            hashed = hashlib.sha256(base_material).digest()
            key = base64.urlsafe_b64encode(hashed)

            try:
                app_config.set("secret.encryption_key", key.decode('utf-8'), is_secret=True)
            except Exception as e:
                logger.error(f"Failed to persist encryption key: {e}")
        else:
            try:
                key = key_str.encode('utf-8')
                Fernet(key)
            except Exception:
                logger.warning(
                    "The configured secret.encryption_key is invalid — deriving a "
                    "fallback key from its hash. Previously encrypted data using the "
                    "original key will become undecryptable. Fix the key in "
                    "ag_system.configs to restore normal operation."
                )
                hashed = hashlib.sha256(key_str.encode('utf-8')).digest()
                key = base64.urlsafe_b64encode(hashed)

        _fernet_instance = Fernet(key)
        return _fernet_instance


def encrypt_secret(plain_text: str) -> str:
    """Encrypt a plain text string.
    
    SECURITY: Raises on failure — NEVER returns plaintext.
    Storing unencrypted secrets in the secret_enc column is a data leak.
    """
    if not plain_text:
        return ""
    f = _get_fernet()
    return f.encrypt(plain_text.encode('utf-8')).decode('utf-8')


def _looks_like_fernet(token: str) -> bool:
    """A Fernet token is urlsafe-base64 of a blob whose first byte is 0x80,
    which always encodes to the 'gAAAAA' prefix. Anything else is not ciphertext
    we produced."""
    return token.startswith("gAAAAA")


def decrypt_secret(encrypted_text: str) -> str:
    """Decrypt an encrypted string.

    Fail-closed: if a value LOOKS like a Fernet token but will not decrypt, the
    encryption key is wrong or has rotated — we RAISE rather than hand the raw
    ciphertext back as if it were the secret (which would ship a useless blob to
    a third-party API and mask a key mismatch). A value that is NOT a Fernet
    token is treated as genuine legacy plaintext and passed through (with a warning
    in production, since prod secrets are expected to be encrypted).
    """
    if not encrypted_text:
        return ""
    try:
        f = _get_fernet()
        return f.decrypt(encrypted_text.encode('utf-8')).decode('utf-8')
    except Exception as e:
        if _looks_like_fernet(encrypted_text):
            logger.error(
                "decrypt_secret: stored ciphertext failed to decrypt — the "
                "encryption key is wrong or has rotated. Refusing to return "
                "ciphertext as a secret."
            )
            raise RuntimeError(
                "Failed to decrypt stored secret: encryption key mismatch"
            ) from e
        env = (os.environ.get("AG_ENVIRONMENT", "")
               or os.environ.get("AG_ENV", "")).strip().lower()
        if env in ("prod", "production"):
            logger.warning(
                "decrypt_secret: value is not encrypted (legacy plaintext) in "
                "production — it should be re-saved to encrypt it."
            )
        else:
            logger.warning("decrypt_secret: assuming legacy plaintext (non-prod).")
        return encrypted_text


_FIELD_ENC_PREFIX = "ENC:"


def is_field_encrypted(value: str) -> bool:
    return bool(value) and value.startswith(_FIELD_ENC_PREFIX)


def _field_key() -> bytes:
    """The shared 32-byte AES key MSystem uses (base64 in config/env)."""
    import os as _os
    key_b64 = _os.environ.get("AG_FIELD_ENCRYPTION_KEY", "")
    if not key_b64:
        try:
            from core.config import app_config
            key_b64 = (app_config.get_global("secret.field_encryption_key") or "").strip()
        except Exception:
            key_b64 = ""
    if not key_b64:
        raise ValueError("field_encryption_key not configured (AG_FIELD_ENCRYPTION_KEY / "
                         "secret.field_encryption_key) — required to decrypt MSystem secrets")
    key = base64.b64decode(key_b64)
    if len(key) != 32:
        raise ValueError(f"field encryption key must be 32 bytes, got {len(key)}")
    return key


def decrypt_field(encrypted_value: str, key_override: bytes = None) -> str:
    """Decrypt a MSystem FieldEncryptor 'ENC:...' value (AES-256-GCM).

    Plaintext (no ENC: prefix) passes through unchanged — so dev/superuser
    credentials and prod encrypted ones both work.
    """
    if not encrypted_value or not is_field_encrypted(encrypted_value):
        return encrypted_value
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = key_override or _field_key()
    combined = base64.b64decode(encrypted_value[len(_FIELD_ENC_PREFIX):])
    nonce, ct_and_tag = combined[:12], combined[12:]
    return AESGCM(key).decrypt(nonce, ct_and_tag, None).decode("utf-8")


def encrypt_field(plaintext: str, key_override: bytes = None) -> str:
    """Encrypt to MSystem FieldEncryptor format (for round-trip tests/parity)."""
    if not plaintext or is_field_encrypted(plaintext):
        return plaintext
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = key_override or _field_key()
    nonce = os.urandom(12)
    ct_and_tag = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return _FIELD_ENC_PREFIX + base64.b64encode(nonce + ct_and_tag).decode("ascii")
