import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from .native_mcp import NativeMCP

logger = logging.getLogger(__name__)

class SmtpActionTools:
    """Email sending via standard SMTP protocol.
    Credentials resolved exclusively via Connections (ag_cred). No fallback.
    """

    def __init__(self, credential_registry=None, user_context_resolver=None,
                 app_id: str = None):
        self._registry = credential_registry
        self._user_context_resolver = user_context_resolver
        self._app_id = app_id

    def _get_config(self) -> dict:
        """Resolve SMTP config via Connections. Returns {} when not configured."""
        if not self._registry:
            return {}
        user_id, app_id = None, self._app_id
        if self._user_context_resolver:
            try:
                ctx = self._user_context_resolver()
                if ctx and getattr(ctx, "user_id", None):
                    uid = str(ctx.user_id).split("_")[-1]
                    user_id = int(uid) if uid.isdigit() else None
                if not app_id and ctx and getattr(ctx, "app_id", None):
                    app_id = ctx.app_id
            except Exception:
                pass
        try:
            cred = self._registry.resolve("smtp", user_id=user_id, app_id=app_id)
            config = dict(cred.config)
            config["password"] = cred.secret
            return config
        except Exception as e:
            logger.debug("smtp._get_config: not configured (%s)", e)
            return {}

    def register_tools(self, mcp_server: NativeMCP):
        instance = self

        @mcp_server.tool()
        def send_email_smtp(to: str, subject: str, body: str, is_html: bool = False) -> str:
            """Send an email using standard SMTP.
            
            Args:
                to: Recipient email address
                subject: Email subject
                body: The text or HTML body
                is_html: Set to true if the body contains HTML
            """
            config = instance._get_config()
            server = config.get("server")
            port = config.get("port")
            username = config.get("username")
            password = config.get("password")
            use_tls = config.get("use_tls", True)
            
            if not server or not port or not username or not password:
                return "✗ Send failed: SMTP integration is not fully configured for this user."

            try:
                msg = MIMEMultipart()
                msg['From'] = username
                msg['To'] = to
                msg['Subject'] = subject

                mime_type = 'html' if is_html else 'plain'
                msg.attach(MIMEText(body, mime_type))

                with smtplib.SMTP(server, port, timeout=15) as server_conn:
                    if use_tls:
                        server_conn.starttls()
                    server_conn.login(username, password)
                    server_conn.send_message(msg)

                return f"✓ Email sent successfully via SMTP to {to}"
            except Exception as e:
                return f"✗ SMTP Send Error: {str(e)}"
