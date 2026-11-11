import logging
from bs4 import BeautifulSoup
from typing import Optional

from brain.models import DocumentType
from brain.ingestion.parsers.base import ParsedDocument
from brain.ingestion.connectors.microsoft.base import MicrosoftGraphClient

logger = logging.getLogger(__name__)

class TeamsConnector:
    """Connects to Microsoft Teams via Microsoft Graph."""

    def __init__(self):
        self.base_client = MicrosoftGraphClient()

    def ingest_channel(self, team_id: str, channel_id: str, max_results: int = 50, since: str = "") -> list[ParsedDocument]:
        """
        Fetch chat messages from a specific Teams channel.
        """
        client = self.base_client.get_client()
        if not client:
            return []

        documents = []
        try:
            logger.info("teams_connector.fetching_messages: team=%s channel=%s since=%s", team_id, channel_id, since or "all")
            params = f"$top={max_results}"
            if since:
                params += f"&$filter=lastModifiedDateTime gt {since}"
            url = f"/teams/{team_id}/channels/{channel_id}/messages?{params}"
            response = client.get(url)
            
            if response.status_code != 200:
                logger.error("teams_connector.api_error: status=%s body=%s", response.status_code, response.text)
                return []

            data = response.json().get('value', [])

            for msg in data:
                parsed_doc = self._parse_message(msg, team_id, channel_id)
                if parsed_doc:
                    documents.append(parsed_doc)

            logger.info("teams_connector.ingested: count=%s since=%s", len(documents), since or "full_sync")
        except Exception as e:
            logger.error("teams_connector.fetch_failed: error=%s", str(e))

        return documents

    def ingest_chat(self, chat_id: str, max_results: int = 50, since: str = "") -> list[ParsedDocument]:
        """
        Fetch chat messages from a specific personal 1-on-1 or group chat.
        """
        client = self.base_client.get_client()
        if not client:
            return []

        documents = []
        try:
            logger.info("teams_connector.fetching_chat_messages: chat=%s since=%s", chat_id, since or "all")
            params = f"$top={max_results}"
            if since:
                params += f"&$filter=lastModifiedDateTime gt {since}"
            url = f"/chats/{chat_id}/messages?{params}"
            response = client.get(url)
            
            if response.status_code != 200:
                logger.error("teams_connector.api_error: status=%s body=%s", response.status_code, response.text)
                return []

            data = response.json().get('value', [])

            for msg in data:
                parsed_doc = self._parse_message(msg, team_id=None, channel_id=None, chat_id=chat_id)
                if parsed_doc:
                    documents.append(parsed_doc)

            logger.info("teams_connector.chat_ingested: count=%s since=%s", len(documents), since or "full_sync")
        except Exception as e:
            logger.error("teams_connector.fetch_failed: error=%s", str(e))

        return documents

    def _parse_message(self, msg: dict, team_id: Optional[str] = None, channel_id: Optional[str] = None, chat_id: Optional[str] = None) -> Optional[ParsedDocument]:
        msg_id = msg.get('id')
        
        from_field = msg.get('from') or {}
        user_field = from_field.get('user') or {}
        sender_name = user_field.get('displayName', 'Unknown')
        
        date = msg.get('createdDateTime', '')
        
        body_field = msg.get('body') or {}
        body_content = body_field.get('content') or ''
        
        if body_field.get('contentType') == 'html':
            soup = BeautifulSoup(body_content, "html.parser")
            body = soup.get_text(separator="\n", strip=True)
        else:
            body = body_content

        if not body.strip():
            return None

        content_parts = [
            f"Author: {sender_name}",
            f"Date: {date}",
            "\n--- Message ---",
            body
        ]
        
        content = "\n".join(content_parts)

        metadata = {
            "format": "teams_message",
            "message_id": msg_id,
            "author": sender_name,
            "date": date
        }
        
        if team_id and channel_id:
            metadata["team_id"] = team_id
            metadata["channel_id"] = channel_id
            source_path = f"teams://{team_id}/{channel_id}/{msg_id}"
        elif chat_id:
            metadata["chat_id"] = chat_id
            source_path = f"teams_chat://{chat_id}/{msg_id}"
        else:
            source_path = f"teams_unknown://{msg_id}"

        return ParsedDocument(
            content=content,
            source_path=source_path,
            document_type=DocumentType.CHAT_MESSAGE,
            title=f"Teams Chat: {sender_name}",
            metadata=metadata,
            raw_size_bytes=len(content.encode('utf-8'))
        )
