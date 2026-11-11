from __future__ import annotations

"""
Agent Brain — Email Parser
==============================
Parses .eml email files using Python's stdlib `email` module.
Extracts headers (From, To, CC, Subject, Date) and body text.
Handles multipart emails by stripping HTML and extracting plain text.
"""

import email
import re
from email import policy
from email.message import EmailMessage
from pathlib import Path
import logging

from brain.models import DocumentType
from brain.ingestion.parsers.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class EmailParser(BaseParser):
    """
    Parser for .eml email files.
    Extracts sender, recipients, subject, date, and body text.
    """

    SUPPORTED_EXTENSIONS = [".eml", ".msg"]

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> ParsedDocument:
        self._validate_file(file_path)

        raw_bytes = file_path.read_bytes()
        raw_size = len(raw_bytes)

        msg = email.message_from_bytes(raw_bytes, policy=policy.default)

        sender = str(msg.get("From", ""))
        to = str(msg.get("To", ""))
        cc = str(msg.get("Cc", ""))
        subject = str(msg.get("Subject", ""))
        date = str(msg.get("Date", ""))
        message_id = str(msg.get("Message-ID", ""))

        body = self._extract_body(msg)

        content_parts = []
        content_parts.append(f"From: {sender}")
        content_parts.append(f"To: {to}")
        if cc:
            content_parts.append(f"CC: {cc}")
        content_parts.append(f"Subject: {subject}")
        content_parts.append(f"Date: {date}")
        content_parts.append("")
        content_parts.append(body)

        content = "\n".join(content_parts)

        metadata = {
            "format": "email",
            "from": sender,
            "to": to,
            "cc": cc,
            "subject": subject,
            "date": date,
            "message_id": message_id,
            "author": self._extract_email_address(sender),
            "has_attachments": self._has_attachments(msg),
        }

        title = f"Email: {subject}" if subject else f"Email from {sender}"

        logger.info("email_parser.parsed: path=%s subject=%s sender=%s", str(file_path), subject[:50], sender[:50])

        return ParsedDocument(
            content=content,
            source_path=str(file_path),
            document_type=DocumentType.EMAIL,
            title=title,
            metadata=metadata,
            raw_size_bytes=raw_size,
        )

    def _extract_body(self, msg: EmailMessage) -> str:
        """Extract the text body from an email message."""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disposition:
                    continue

                if content_type == "text/plain":
                    payload = part.get_content()
                    if isinstance(payload, str):
                        body = payload
                        break
                elif content_type == "text/html" and not body:
                    payload = part.get_content()
                    if isinstance(payload, str):
                        body = self._strip_html(payload)
        else:
            content_type = msg.get_content_type()
            payload = msg.get_content()
            if isinstance(payload, str):
                if content_type == "text/html":
                    body = self._strip_html(payload)
                else:
                    body = payload

        return body.strip()

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and decode entities to get plain text."""
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("&nbsp;", " ").replace("&amp;", "&")
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_email_address(sender: str) -> str:
        """Extract just the email address from a 'Name <email>' string."""
        match = re.search(r"<([^>]+)>", sender)
        if match:
            return match.group(1)
        return sender.strip()

    @staticmethod
    def _has_attachments(msg: EmailMessage) -> bool:
        """Check if the email has any attachments."""
        if not msg.is_multipart():
            return False
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                return True
        return False
