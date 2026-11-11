"""
Retriever evaluation — precision@k / recall@k / MRR (roadmap C.6).
═══════════════════════════════════════════════════════════════════

Two layers:
  1. Pure metric functions (unit-tested, no infra).
  2. A runnable harness that executes labeled queries against the live
     DocumentStore (hybrid FTS+vector) and reports per-query + aggregate
     metrics. Run on a box with DB access:

        python -m tests.eval.retrieval_eval tests/eval/cases/retrieval_labels.json

Labeled-set format (JSON list):
    [{"query": "how do users authenticate",
      "relevant": ["<document_id>", "..."],     # ground-truth doc ids
      "k": 5}]                                   # optional, default 5

Build the labeled set from real usage: take questions users actually asked
(transcripts), mark which retrieved docs answered them.
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Sequence


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str],
                   k: int) -> float:
    """Fraction of the top-k results that are relevant. 0.0 for empty input."""
    if k <= 0 or not ranked_ids:
        return 0.0
    rel = set(relevant_ids)
    top = list(ranked_ids)[:k]
    return sum(1 for d in top if d in rel) / float(k)


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str],
                k: int) -> float:
    """Fraction of all relevant docs that appear in the top-k."""
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    top = set(list(ranked_ids)[:k])
    return len(rel & top) / float(len(rel))


def mrr(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Reciprocal rank of the FIRST relevant result (0.0 if none)."""
    rel = set(relevant_ids)
    for i, d in enumerate(ranked_ids, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def evaluate_case(ranked_ids: Sequence[str], relevant_ids: Sequence[str],
                  k: int = 5) -> Dict[str, float]:
    return {
        "precision_at_k": precision_at_k(ranked_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(ranked_ids, relevant_ids, k),
        "mrr": mrr(ranked_ids, relevant_ids),
    }


def aggregate(per_case: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across cases."""
    if not per_case:
        return {}
    keys = per_case[0].keys()
    return {key: sum(c[key] for c in per_case) / len(per_case) for key in keys}


def run_live(labels_path: str, k_default: int = 5) -> Dict:
    """Execute labeled queries against the live hybrid retriever."""
    from core.tenant_db import TenantDB
    from brain.storage.document_store import DocumentStore
    from brain.retrieval.search_agent import SearchAgent

    with open(labels_path) as f:
        cases = json.load(f)

    store = DocumentStore(TenantDB())
    agent = SearchAgent(lexical_store=None, document_store=store)

    results, per_case = [], []
    for case in cases:
        k = int(case.get("k", k_default))
        docs = agent._hybrid_fetch(case["query"], limit=k,
                                   semantic_query=case.get("semantic_query"))
        ranked = [str(d.id) for d in docs]
        metrics = evaluate_case(ranked, [str(r) for r in case["relevant"]], k)
        per_case.append(metrics)
        results.append({"query": case["query"], "ranked": ranked, **metrics})

    summary = {"cases": results, "aggregate": aggregate(per_case)}
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out = run_live(sys.argv[1])
    print(json.dumps(out, indent=2))
    agg = out["aggregate"]
    print(f"\nAggregate: P@k={agg.get('precision_at_k', 0):.2f} "
          f"R@k={agg.get('recall_at_k', 0):.2f} MRR={agg.get('mrr', 0):.2f}")
