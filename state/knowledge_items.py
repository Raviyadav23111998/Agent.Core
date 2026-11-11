import os
import json
from typing import List, Optional


class KnowledgeItemManager:
    """Repository-specific context loading system.
    
    Architecture (identical to Agent):
    - At init: loads ONLY summaries from metadata.json (lightweight)
    - During execution: loads full artifact content on-demand when needed
    - Prevents unnecessary context bloat from unused KIs
    - KIs are starting points, not ground truth — agent verifies against code
    
    Directory structure:
    .knowledge/
    ├── ki_001/
    │   ├── metadata.json    # {"summary": "...", "primary_artifact": "patterns.md"}
    │   └── artifacts/
    │       └── patterns.md  # Full detailed context
    └── ki_002/
        ├── metadata.json
        └── artifacts/
            └── api_guide.md
    """
    
    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)
        self.ki_dir = os.path.join(self.workspace_root, '.knowledge')
        self._ki_index: List[dict] = []
        self._build_index()
        
    def _build_index(self):
        """Build lightweight index of all KIs (summaries only, no full content)."""
        if not os.path.exists(self.ki_dir):
            return
            
        for root, dirs, files in os.walk(self.ki_dir):
            if 'metadata.json' in files:
                try:
                    meta_path = os.path.join(root, 'metadata.json')
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    artifact_path = os.path.join(root, data.get('primary_artifact', ''))
                    
                    self._ki_index.append({
                        "summary": data.get('summary', 'Unknown KI'),
                        "artifact_path": artifact_path,
                        "metadata_path": meta_path,
                        "ki_dir": root,
                        "loaded": False,
                        "content": None,
                    })
                except Exception:
                    continue
    
    def load_active_kis(self) -> List[str]:
        """Return lightweight KI summaries for system prompt injection.
        
        This is called at init — returns summaries ONLY, not full content.
        """
        return [
            f"KI: {ki['summary']}. Details: {ki['artifact_path']}"
            for ki in self._ki_index
        ]
    
    def load_ki_content(self, ki_index: int) -> Optional[str]:
        """Load full content of a specific KI on-demand.
        
        Called when the agent determines a KI is relevant to the current task.
        """
        if ki_index < 0 or ki_index >= len(self._ki_index):
            return None
        
        ki = self._ki_index[ki_index]
        
        if ki["loaded"]:
            return ki["content"]
        
        if not os.path.exists(ki["artifact_path"]):
            return None
        
        try:
            with open(ki["artifact_path"], 'r', encoding='utf-8') as f:
                content = f.read()
            ki["content"] = content
            ki["loaded"] = True
            return content
        except Exception:
            return None
    
    def search_kis(self, query: str) -> List[dict]:
        """Search KI summaries by keyword overlap.
        
        Returns matching KIs with their index for on-demand loading.
        """
        query_tokens = set(query.lower().split())
        results = []
        
        for i, ki in enumerate(self._ki_index):
            summary_tokens = set(ki["summary"].lower().split())
            overlap = query_tokens & summary_tokens
            if overlap:
                score = len(overlap) / len(query_tokens) if query_tokens else 0
                results.append({
                    "index": i,
                    "summary": ki["summary"],
                    "score": score,
                    "path": ki["artifact_path"],
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def get_ki_count(self) -> int:
        """Return number of indexed KIs."""
        return len(self._ki_index)
