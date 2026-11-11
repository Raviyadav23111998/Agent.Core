import requests
import json
from .native_mcp import NativeMCP

class DatabaseTools:
    """
    Direct, dependency-free database integration using REST APIs.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str, es_url: str, es_index: str):
        import os
        # NEO4J_HTTP_URL lets a deployment whose Neo4j is not on the default
        # ports (e.g. a dedicated instance on 7688/7475) point this at its HTTP API.
        http_base = (os.environ.get("NEO4J_HTTP_URL", "").strip().rstrip("/")
                     or neo4j_uri.replace("bolt://", "http://").replace(":7687", ":7474"))
        self.neo4j_url = http_base + "/db/neo4j/tx/commit"
        self.neo4j_auth = (neo4j_user, neo4j_pass)
        self.es_url = f"{es_url}/{es_index}/_search"
        
    def register_tools(self, mcp_server: NativeMCP):
        
        @mcp_server.tool()
        def query_neo4j(cypher_query: str) -> str:
            """Executes raw Cypher on the Neo4j Knowledge Graph."""
            payload = {
                "statements": [
                    {"statement": cypher_query}
                ]
            }
            try:
                resp = requests.post(self.neo4j_url, auth=self.neo4j_auth, json=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("errors"):
                    return f"Cypher Error: {json.dumps(data['errors'])}"
                    
                results = data.get("results", [])
                if not results: return "No results."
                
                output = []
                for row in results[0].get("data", []):
                    output.append(json.dumps(row.get("row")))
                
                return "\n".join(output) if output else "Query succeeded (no records returned)."
            except Exception as e:
                return f"Neo4j Connection Error: {str(e)}"

        @mcp_server.tool()
        def search_elasticsearch(query: str) -> str:
            """Executes a lexical text search against Elasticsearch."""
            payload = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["*"]
                    }
                },
                "size": 5
            }
            try:
                resp = requests.post(self.es_url, json=payload, timeout=10)
                resp.raise_for_status()
                hits = resp.json().get("hits", {}).get("hits", [])
                
                if not hits: return "No matches found."
                
                output = []
                for hit in hits:
                    source = hit.get("_source", {})
                    summary = json.dumps(source)[:500] 
                    output.append(f"Score: {hit.get('_score')} | Data: {summary}")
                    
                return "\n".join(output)
            except Exception as e:
                return f"Elasticsearch Error: {str(e)}"
