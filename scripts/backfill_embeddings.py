#!/usr/bin/env python3
"""
Backfill dense embeddings for already-ingested brain documents.

New documents are embedded at ingestion (StorageManager.ingest_document).
This script embeds the back catalogue after an embedding model is first
configured (or changed). Idempotent: only touches rows where embedding IS
NULL, so it is safe to re-run and to interrupt.

Usage (from Agent/agent.core):
    python -m scripts.backfill_embeddings            # all apps
    python -m scripts.backfill_embeddings --app 1    # one app
    python -m scripts.backfill_embeddings --batch 32 --limit 1000

Requires an embedding model configured (llm.embedding_model / _base_url /
_api_key); exits early with a clear message otherwise.
"""
import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill brain document embeddings.")
    ap.add_argument("--app", type=int, default=None, help="Restrict to one app_id.")
    ap.add_argument("--batch", type=int, default=16, help="Embeddings per API call.")
    ap.add_argument("--limit", type=int, default=100000, help="Max docs this run.")
    args = ap.parse_args()

    from core.config import app_config
    app_config.load()

    from core.tenant_db import TenantDB
    from brain.storage.document_store import DocumentStore
    from core.embeddings import EmbeddingClient

    emb = EmbeddingClient()
    if not emb.available:
        print("✗ No embedding model configured "
              "(set llm.embedding_model / llm.embedding_base_url / "
              "llm.embedding_api_key). Nothing to do.")
        return 1

    db = TenantDB()
    db.connect()
    store = DocumentStore(db)

    stats = store.count_embedded(app_id=args.app)
    pending = stats["total"] - stats["embedded"]
    print(f"Documents: {stats['total']} total, {stats['embedded']} embedded, "
          f"{pending} to backfill. Model: {emb.model}")

    done = 0
    while done < args.limit:
        docs = store.get_documents_without_embedding(
            limit=min(args.batch, args.limit - done), app_id=args.app)
        if not docs:
            break
        texts = [(f"{d.title}\n\n{d.content}" if d.title else d.content)[:8000]
                 for d in docs]
        vectors = emb.embed(texts)
        if not vectors or len(vectors) != len(docs):
            print(f"  ! embedding batch returned {len(vectors)} for {len(docs)} "
                  f"docs — stopping to avoid misalignment.")
            break
        for d, v in zip(docs, vectors):
            store.update_embedding(d.id, v, model=emb.model or "")
            done += 1
        print(f"  embedded {done} (last doc_id={docs[-1].id})")
        time.sleep(0.1)

    final = store.count_embedded(app_id=args.app)
    print(f"✓ Done. {final['embedded']}/{final['total']} documents embedded.")

    from brain.ingestion.chunker import chunk_text
    cstats = store.count_chunks(app_id=args.app)
    print(f"Chunks: {cstats['docs_with_chunks']} docs already chunked, "
          f"{cstats['embedded_chunks']} embedded chunks.")
    try:
        csize = app_config.get_int("brain.chunk_size", 2000)
        coverlap = app_config.get_int("brain.chunk_overlap", 320)
    except Exception:
        csize, coverlap = 2000, 320

    chunked_docs = 0
    while chunked_docs < args.limit:
        docs = store.get_documents_without_chunks(
            limit=min(args.batch, args.limit - chunked_docs), app_id=args.app)
        if not docs:
            break
        for d in docs:
            body = f"{d.title}\n\n{d.content}" if d.title else d.content
            chunks = chunk_text(body, chunk_size=csize, overlap=coverlap)
            if not chunks:
                store.replace_chunks(d.id, [], model=emb.model or "")
                chunked_docs += 1
                continue
            texts = [c.content for c in chunks]
            vecs = []
            for i in range(0, len(texts), 32):
                vecs.extend(emb.embed(texts[i:i + 32]) or [])
            if len(vecs) != len(texts):
                vecs = [None] * len(texts)
            payload = [{"content": c.content, "embedding": v}
                       for c, v in zip(chunks, vecs)]
            store.replace_chunks(d.id, payload, model=emb.model or "")
            chunked_docs += 1
        print(f"  chunked {chunked_docs} docs (last doc_id={docs[-1].id})")
        time.sleep(0.1)

    fc = store.count_chunks(app_id=args.app)
    print(f"✓ Chunking done. {fc['docs_with_chunks']} docs chunked, "
          f"{fc['embedded_chunks']} embedded chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
