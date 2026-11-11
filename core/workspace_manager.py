"""
Workspace Manager — Per-User Sandboxed Workspace Provisioning
═══════════════════════════════════════════════════════════════════

Architecture (Google Drive model):
    /var/agent-workspaces/
    └── {tenant_id}/
        ├── user_1/            # Admin
        │   ├── projects/
        │   ├── uploads/
        │   └── .workspace.json
        ├── user_5/            # Regular user  
        │   ├── projects/
        │   ├── uploads/
        │   └── .workspace.json
        └── __shared__/        # Read-only shared resources (future)

Security:
    - Each user gets a physically isolated directory
    - FilesystemTools + ShellExecutionTool are jailed to this directory
    - Path traversal blocked by realpath + boundary checks
    - Disk quotas configurable per user via DB (ag_system.configs)
"""

import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACES_ROOT = "/var/agent-workspaces"

_DEFAULT_SUBDIRS = ["projects", "uploads"]


class WorkspaceManager:
    """Provisions and manages per-user workspace directories.
    
    Each user gets their own directory:
        {workspaces_root}/{tenant_id}/user_{user_id}/
    
    Configuration loaded from ag_system.configs:
        workspace.root           — Root directory (default: /var/agent-workspaces)
        workspace.default_quota_mb — Default quota in MB (default: 500)
        workspace.tenant_id      — Tenant ID (default: default)
        workspace.quota.user_{id} — Per-user quota override in MB
    """
    
    def __init__(self, workspaces_root: str = None, tenant_id: str = "default"):
        self._config = self._load_config()
        
        self._root = os.path.abspath(
            workspaces_root or 
            self._config.get("workspace.root", DEFAULT_WORKSPACES_ROOT)
        )
        self._tenant_id = tenant_id or self._config.get("workspace.tenant_id", "default")
        self._tenant_root = os.path.join(self._root, self._tenant_id)
        self._default_quota_mb = int(self._config.get("workspace.default_quota_mb", 500))
    
    @staticmethod
    def _load_config() -> dict:
        """Load workspace config from DB. Fails silently if DB unavailable."""
        try:
            from core.config import app_config
            return {
                "workspace.root": app_config.get("workspace.root", DEFAULT_WORKSPACES_ROOT),
                "workspace.default_quota_mb": app_config.get("workspace.default_quota_mb", "500"),
                "workspace.tenant_id": app_config.get("workspace.tenant_id", "default"),
            }
        except Exception:
            return {}
    
    def get_user_quota_bytes(self, user_id: int) -> int:
        """Get workspace quota for a user in bytes.
        
        Checks for per-user override (workspace.quota.user_{id}), 
        falls back to default_quota_mb from config.
        Returns 0 if no quota configured (unlimited).
        """
        try:
            from core.config import app_config
            per_user = app_config.get(f"workspace.quota.user_{user_id}")
            if per_user:
                return int(per_user) * 1024 * 1024
        except Exception:
            pass
        
        return self._default_quota_mb * 1024 * 1024
    
    @property
    def tenant_root(self) -> str:
        """Root directory for this tenant's workspaces."""
        return self._tenant_root
    
    def get_user_workspace(self, user_id: int) -> str:
        """Get the workspace path for a user (may not exist yet).
        
        Returns:
            Absolute path: {root}/{tenant}/user_{id}/
        """
        return os.path.join(self._tenant_root, f"user_{user_id}")
    
    def ensure_workspace(self, user_id: int) -> str:
        """Get or create the workspace directory for a user.
        
        Creates the directory structure and metadata file if it doesn't exist.
        Idempotent — safe to call on every request.
        
        Returns:
            Absolute path to the user's workspace directory.
        """
        ws_path = self.get_user_workspace(user_id)
        
        if os.path.isdir(ws_path):
            return ws_path
        
        os.makedirs(ws_path, mode=0o700, exist_ok=True)
        
        for subdir in _DEFAULT_SUBDIRS:
            os.makedirs(os.path.join(ws_path, subdir), mode=0o700, exist_ok=True)
        
        meta_path = os.path.join(ws_path, ".workspace.json")
        if not os.path.exists(meta_path):
            meta = {
                "user_id": user_id,
                "tenant_id": self._tenant_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": 1,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        
        logger.info("workspace.created: user=%d path=%s", user_id, ws_path)
        return ws_path
    
    def workspace_exists(self, user_id: int) -> bool:
        """Check if a user's workspace has been provisioned."""
        return os.path.isdir(self.get_user_workspace(user_id))
    
    def get_workspace_size(self, user_id: int) -> int:
        """Calculate total size of a user's workspace in bytes.
        
        Used for quota enforcement.
        """
        ws_path = self.get_user_workspace(user_id)
        if not os.path.isdir(ws_path):
            return 0
        
        total = 0
        for dirpath, _, filenames in os.walk(ws_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total
    
    def get_workspace_info(self, user_id: int) -> dict:
        """Get workspace metadata and usage stats."""
        ws_path = self.get_user_workspace(user_id)
        
        if not os.path.isdir(ws_path):
            return {"exists": False, "user_id": user_id}
        
        file_count = 0
        for _, _, files in os.walk(ws_path):
            file_count += len(files)
        
        size_bytes = self.get_workspace_size(user_id)
        
        meta = {}
        meta_path = os.path.join(ws_path, ".workspace.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {
            "exists": True,
            "user_id": user_id,
            "path": ws_path,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "file_count": file_count,
            "created_at": meta.get("created_at"),
        }
    
    def list_all_workspaces(self) -> list:
        """List all user workspaces under this tenant (admin use)."""
        if not os.path.isdir(self._tenant_root):
            return []
        
        workspaces = []
        for name in sorted(os.listdir(self._tenant_root)):
            if name.startswith("user_"):
                try:
                    uid = int(name.split("_", 1)[1])
                    workspaces.append(self.get_workspace_info(uid))
                except (ValueError, IndexError):
                    pass
        return workspaces
