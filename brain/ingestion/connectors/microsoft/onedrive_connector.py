import tempfile
import logging
from pathlib import Path

from brain.ingestion.parsers.base import ParsedDocument
from brain.ingestion.connectors.file_connector import FileConnector
from brain.ingestion.connectors.microsoft.base import MicrosoftGraphClient

logger = logging.getLogger(__name__)

class OneDriveConnector:
    """Connects to Microsoft OneDrive/SharePoint via Microsoft Graph."""

    def __init__(self):
        self.base_client = MicrosoftGraphClient()
        self.file_connector = FileConnector()

    def ingest_folder(self, folder_path: str = "root/children", max_results: int = 50, since: str = "") -> list[ParsedDocument]:
        """
        Fetch files from OneDrive.
        Downloads them temporarily and delegates parsing to FileConnector.
        """
        client = self.base_client.get_client()
        if not client:
            return []

        documents = []
        try:
            logger.info("onedrive_connector.fetching_files: folder=%s since=%s", folder_path, since or "all")
            params = f"$top={max_results}"
            if since:
                params += f"&$filter=lastModifiedDateTime gt {since}"
            url = f"/me/drive/{folder_path}?{params}"
            response = client.get(url)
            
            if response.status_code != 200:
                logger.error("onedrive_connector.api_error: status=%s body=%s", response.status_code, response.text)
                return []

            data = response.json().get('value', [])
            
            files = [item for item in data if 'file' in item]

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                
                for file_item in files:
                    file_id = file_item.get('id')
                    file_name = file_item.get('name')
                    download_url = file_item.get('@microsoft.graph.downloadUrl')
                    date = file_item.get('lastModifiedDateTime', '')
                    
                    if not download_url:
                        continue
                        
                    download_response = client.get(download_url)
                    if download_response.status_code == 200:
                        local_file = tmp_path / file_name
                        local_file.write_bytes(download_response.content)
                        
                        parsed_doc = self.file_connector.parse_file(local_file)
                        
                        if parsed_doc and not parsed_doc.is_empty:
                            parsed_doc.source_path = f"onedrive://{file_id}/{file_name}"
                            parsed_doc.metadata["format"] = "onedrive_file"
                            parsed_doc.metadata["date"] = date
                            documents.append(parsed_doc)
                            
                        local_file.unlink()

            logger.info("onedrive_connector.ingested: count=%s since=%s", len(documents), since or "full_sync")
        except Exception as e:
            logger.error("onedrive_connector.fetch_failed: error=%s", str(e))

        return documents
