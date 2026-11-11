"""
Agent Brain — Package Init
========================================
NLP constants shared across all retrieval agents.
"""

NLP_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "where", "when", "how", "why",
    "and", "or", "but", "not", "so", "if", "then", "than",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "all", "any", "each", "every", "some", "no",
})
