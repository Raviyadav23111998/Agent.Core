import imaplib
import email
import email.policy
import logging
from .native_mcp import NativeMCP

logger = logging.getLogger(__name__)

class ImapActionTools:
    """Email fetching via standard IMAP protocol.
    Credentials resolved exclusively via Connections (ag_cred). No fallback.
    """

    def __init__(self, credential_registry=None, user_context_resolver=None,
                 app_id: str = None):
        self._registry = credential_registry
        self._user_context_resolver = user_context_resolver
        self._app_id = app_id

    def _get_config(self) -> dict:
        """Resolve IMAP config via Connections. Returns {} when not configured."""
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
            cred = self._registry.resolve("imap", user_id=user_id, app_id=app_id)
            config = dict(cred.config)
            config["password"] = cred.secret
            return config
        except Exception as e:
            logger.debug("imap._get_config: not configured (%s)", e)
            return {}

    def register_tools(self, mcp_server: NativeMCP):
        instance = self

        @mcp_server.tool()
        def fetch_emails_imap(folder: str = "INBOX", max_count: int = 5, unread_only: bool = False) -> str:
            """Fetch recent emails via IMAP protocol (use this for non-M365 mailboxes).
            
            Args:
                folder: The mail folder to fetch from (default: INBOX)
                max_count: Maximum number of emails to fetch (default: 5)
                unread_only: If True, only fetches unread (UNSEEN) emails
            """
            config = instance._get_config()
            server = config.get("server")
            port = config.get("port", 993)
            username = config.get("username")
            password = config.get("password")
            use_ssl = config.get("use_ssl", True)
            
            if not server or not username or not password:
                return "✗ Fetch failed: IMAP integration is not fully configured for this user."

            try:
                if use_ssl:
                    mail = imaplib.IMAP4_SSL(server, port)
                else:
                    mail = imaplib.IMAP4(server, port)
                
                mail.login(username, password)
                mail.select(folder)

                search_criteria = "UNSEEN" if unread_only else "ALL"
                status, response = mail.search(None, search_criteria)
                
                if status != "OK":
                    return f"✗ IMAP Search Error: Failed to search folder {folder}"

                msg_ids = response[0].split()
                if not msg_ids:
                    return "No emails found matching the criteria."

                recent_ids = msg_ids[-max_count:]
                
                emails_output = []
                for msg_id in reversed(recent_ids):
                    status, msg_data = mail.fetch(msg_id, "(RFC822)")
                    if status != "OK":
                        continue
                        
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1], policy=email.policy.default)
                            subject = msg.get("subject", "(No Subject)")
                            sender = msg.get("from", "(Unknown Sender)")
                            date = msg.get("date", "(Unknown Date)")
                            
                            body = ""
                            if msg.is_multipart():
                                for part in msg.walk():
                                    ctype = part.get_content_type()
                                    cdispo = str(part.get('Content-Disposition'))
                                    
                                    if ctype == 'text/plain' and 'attachment' not in cdispo:
                                        body = part.get_content()
                                        break
                                if not body:
                                    for part in msg.walk():
                                        if part.get_content_type() == 'text/html':
                                            body = part.get_content()
                                            break
                            else:
                                body = msg.get_content()

                            if not body:
                                body = "(No content)"
                            body_str = str(body)
                            body_preview = body_str[:500] + ("..." if len(body_str) > 500 else "")
                            
                            emails_output.append(
                                f"From: {sender}\n"
                                f"Date: {date}\n"
                                f"Subject: {subject}\n"
                                f"Body Preview:\n{body_preview}\n"
                                f"-" * 40
                            )

                mail.close()
                mail.logout()

                if not emails_output:
                    return "No emails could be fetched."

                return "\n".join(emails_output)
            except Exception as e:
                return f"✗ IMAP Fetch Error: {str(e)}"
