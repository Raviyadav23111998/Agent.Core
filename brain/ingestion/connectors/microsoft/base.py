import time
import threading
import logging
from typing import Optional
from azure.identity import DeviceCodeCredential, TokenCachePersistenceOptions, ClientSecretCredential
import httpx

logger = logging.getLogger(__name__)


class HeadlessAuthRequired(Exception):
    """Raised when Device Code auth is needed but we're running headlessly."""
    pass


class MicrosoftGraphClient:
    """
    Base client for connecting to Microsoft Graph.
    Supports both Client Credentials (App-Only) and Device Code (Delegated) flows.
    Implements automatic token refresh when tokens expire.
    
    Credentials resolved via CredentialRegistry (vault + bindings) — the
    single source of truth for the m365 app registration.
    """
    _instance = None
    _lock = threading.Lock()
    _credential_registry = None

    @classmethod
    def set_credential_registry(cls, registry):
        """Set the CredentialRegistry for credential resolution."""
        cls._credential_registry = registry

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MicrosoftGraphClient, cls).__new__(cls)
                cls._instance._credential = None
                cls._instance._client = None
                cls._instance._token = None
                cls._instance._token_expiry = 0
                cls._instance._is_app_only = False
            return cls._instance

    def _get_fresh_token(self) -> str:
        """Get a valid token, refreshing if expired or about to expire."""
        now = time.time()
        if self._token and (now < self._token_expiry - 300):
            return self._token.token

        if not self._credential:
            raise RuntimeError("Credential not initialized. Call connect() first.")

        import concurrent.futures
        scopes = (
            ("https://graph.microsoft.com/.default",)
            if self._is_app_only
            else (
                "https://graph.microsoft.com/User.Read",
                "https://graph.microsoft.com/Mail.Read",
                "https://graph.microsoft.com/Files.Read.All",
                "https://graph.microsoft.com/ChannelMessage.Read.All",
                "https://graph.microsoft.com/Team.ReadBasic.All",
                "https://graph.microsoft.com/Channel.ReadBasic.All",
                "https://graph.microsoft.com/Group.Read.All",
                "https://graph.microsoft.com/Chat.Read",
            )
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._credential.get_token, *scopes)
            try:
                self._token = future.result(timeout=10)
            except concurrent.futures.TimeoutError:
                raise RuntimeError("Azure AD token acquisition timed out after 10s — check network and credentials")

        self._token_expiry = self._token.expires_on
        logger.info("microsoft_graph.token_refreshed: expires_in=%s", int(self._token_expiry - now))
        return self._token.token

    def connect(self) -> Optional[httpx.Client]:
        """Initialize the credential and create the HTTP client."""
        if self._credential:
            try:
                token_str = self._get_fresh_token()
                self._client = httpx.Client(
                    base_url="https://graph.microsoft.com/v1.0",
                    headers={"Authorization": f"Bearer {token_str}"},
                    timeout=30.0,
                )
                return self._client
            except Exception as e:
                logger.error("microsoft_graph.token_refresh_failed: error=%s", str(e))
                return None

        tenant_id = None
        client_id = None
        client_secret = None

        if self._credential_registry:
            try:
                cred = self._credential_registry.resolve("m365")
                tenant_id = cred.config.get("tenant_id")
                client_id = cred.config.get("client_id")
                client_secret = cred.secret
            except Exception as e:
                logger.debug("microsoft_graph.resolve_failed: %s", e)

        if not tenant_id or not client_id:
            logger.warning("microsoft_graph.missing_credentials: no m365 app credential (tenant_id/client_id) in the vault")
            return None

        try:
            if client_secret:
                self._credential = ClientSecretCredential(
                    client_id=client_id,
                    tenant_id=tenant_id,
                    client_secret=client_secret,
                )
                self._is_app_only = True
                token_str = self._get_fresh_token()
                logger.info("microsoft_graph.connected: msg=%s", "Successfully initialized via ClientSecretCredential")
            else:
                cache_options = TokenCachePersistenceOptions(
                    name="companybrain_ms_cache",
                    allow_unencrypted_storage=True,
                )

                def headless_prompt_callback(uri, code, expires_on):
                    raise HeadlessAuthRequired(
                        f"⚠️ Microsoft Authentication Required! Please go to {uri} and enter code: {code}"
                    )

                self._credential = DeviceCodeCredential(
                    client_id=client_id,
                    tenant_id=tenant_id,
                    cache_persistence_options=cache_options,
                    prompt_callback=headless_prompt_callback,
                )
                self._is_app_only = False
                token_str = self._get_fresh_token()
                logger.info("microsoft_graph.connected: msg=%s", "Successfully initialized via DeviceCodeCredential")

            self._client = httpx.Client(
                base_url="https://graph.microsoft.com/v1.0",
                headers={"Authorization": f"Bearer {token_str}"},
                timeout=30.0,
            )

            return self._client
        except Exception as e:
            logger.error("microsoft_graph.connection_failed: error=%s", str(e))
            return None

    def get_client(self) -> Optional[httpx.Client]:
        """Get an HTTP client with a guaranteed fresh token."""
        if self._credential:
            try:
                token_str = self._get_fresh_token()
                if self._client:
                    self._client.headers["Authorization"] = f"Bearer {token_str}"
                    return self._client
            except Exception as e:
                logger.warning("microsoft_graph.refresh_failed_reconnecting: error=%s", str(e))
                self._credential = None
                self._client = None

        return self.connect()
