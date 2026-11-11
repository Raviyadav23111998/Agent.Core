from __future__ import annotations

"""
Agent Brain — Extraction Prompts
====================================
LLM prompt templates for extracting entities and relations
from different document types.

Each prompt instructs the LLM to return a JSON object with:
  { "entities": [...], "relations": [...] }

Prompts are tuned per document type because:
- Emails: focus on people, communication patterns, topics
- Policies: focus on rules, departments, approvers, compliance
- Code: focus on modules, dependencies, classes, APIs
- Documents: general entity extraction
"""


EXTRACTION_SYSTEM_PROMPT = """Extract entities and relationships from documents for a corporate knowledge graph.

RULES:
1. Extract ONLY factual entities/relations from the text. No hallucination.
2. Entity types: person, organization, project, technology, file, module, policy, concept, event, location
3. Relation types: AUTHORED, EMAILED, MENTIONED_IN, DEPENDS_ON, PART_OF, WORKS_ON, REPORTED, APPROVED, MODIFIED, REFERENCES, RELATED_TO
4. Max 15 entities + 15 relations. Assign confidence 0.0-1.0. Normalize names to lowercase.
5. Return ONLY JSON: {"entities":[{"name":"...","type":"...","properties":{},"confidence":0.9}],"relations":[{"source":"...","target":"...","type":"...","properties":{},"confidence":0.9}]}"""


EMAIL_EXTRACTION_PROMPT = """Extract entities and relationships from this EMAIL.

Focus on:
- WHO sent the email (person entities)
- WHO received it (person entities)
- WHAT topics/projects are discussed (project/concept entities)
- WHAT technologies are mentioned (technology entities)
- WHO is assigned to do WHAT (WORKS_ON relations)
- WHO reported or escalated WHAT (REPORTED relations)

EMAIL CONTENT:
{content}"""


POLICY_EXTRACTION_PROMPT = """Extract entities and relationships from this COMPANY POLICY document.

Focus on:
- WHAT policies/rules are defined (policy entities)
- WHO approved or authored the policy (person entities)
- WHICH departments does it apply to (organization entities)
- WHAT technologies or systems does it reference (technology entities)
- WHO is responsible for enforcement (APPROVED, WORKS_ON relations)
- What does this policy REFERENCE or DEPEND_ON (REFERENCES, DEPENDS_ON relations)

POLICY CONTENT:
{content}"""


CODE_EXTRACTION_PROMPT = """Extract entities and relationships from this SOURCE CODE file.

Focus on:
- WHAT classes/interfaces/services are defined (module entities)
- WHAT external dependencies are imported (technology entities)
- WHAT other files/modules does this code depend on (DEPENDS_ON relations)
- WHAT is the module/feature area (project/concept entities)
- WHO authored this (if indicated by comments/headers) (person entities)

SOURCE CODE ({language}):
{content}"""


DOCUMENT_EXTRACTION_PROMPT = """Extract entities and relationships from this DOCUMENT.

Focus on:
- WHO are the key people mentioned (person entities)
- WHAT projects or initiatives are discussed (project entities)
- WHAT organizations or departments are referenced (organization entities)
- WHAT technologies or tools are mentioned (technology entities)
- WHAT events are described (event entities)
- HOW do the entities relate to each other (all relation types)

DOCUMENT CONTENT:
{content}"""


MEETING_NOTES_EXTRACTION_PROMPT = """Extract entities and relationships from these MEETING NOTES.

Focus on:
- WHO attended the meeting (person entities)
- WHAT was discussed (project/concept entities)
- WHAT action items were assigned and to WHOM (WORKS_ON relations)
- WHAT decisions were made and WHO approved them (APPROVED relations)
- WHAT deadlines or events were mentioned (event entities)

MEETING NOTES:
{content}"""


EXTRACTION_PROMPTS = {
    "email": EMAIL_EXTRACTION_PROMPT,
    "policy": POLICY_EXTRACTION_PROMPT,
    "code": CODE_EXTRACTION_PROMPT,
    "document": DOCUMENT_EXTRACTION_PROMPT,
    "meeting_notes": MEETING_NOTES_EXTRACTION_PROMPT,
    "chat_message": DOCUMENT_EXTRACTION_PROMPT,
    "ticket": DOCUMENT_EXTRACTION_PROMPT,
}


def get_extraction_prompt(document_type: str, content: str, **kwargs) -> str:
    """
    Get the appropriate extraction prompt for a document type.

    Args:
        document_type: The type of document (email, policy, code, etc.)
        content: The document content to inject into the prompt
        **kwargs: Additional template variables (e.g., language for code)

    Returns:
        Formatted prompt string ready to send to the LLM
    """
    template = EXTRACTION_PROMPTS.get(document_type, DOCUMENT_EXTRACTION_PROMPT)
    return template.format(content=content, **kwargs)
