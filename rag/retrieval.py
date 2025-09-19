# rag/retrieval.py

# 1) Try to import Sentence-Transformers and CrossEncoder
HAS_ST = True
try:
    from sentence_transformers import SentenceTransformer, util, CrossEncoder
except Exception:
    HAS_ST = False

# 2) Try BM25; if missing, we’ll do a naive fallback
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_RERANK_MODEL = "BAAI/bge-reranker-base"  # or: "cross-encoder/ms-marco-MiniLM-L-6-v2"

if HAS_ST:
    _embedder = SentenceTransformer(_EMBED_MODEL)
    _reranker = CrossEncoder(_RERANK_MODEL)

def _bm25_topk(query: str, docs: list[str], k: int) -> list[int]:
    if not HAS_BM25 or not docs:
        return []
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    return sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]

def _naive_topk(query: str, docs: list[str], k: int) -> list[int]:
    # simple keyword count fallback
    qwords = query.lower().split()
    scores = [sum(d.lower().count(w) for w in qwords) for d in docs]
    return sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]

def hybrid_retrieve(query: str, docs: list[str], top_k: int = 3) -> list[str]:
    if not docs:
        return []

    # If ST available, do full hybrid (BM25 + dense + rerank)
    if HAS_ST:
        bm25_top = _bm25_topk(query, docs, top_k)
        import torch
        D = _embedder.encode(docs, convert_to_tensor=True, normalize_embeddings=True)
        q = _embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        from sentence_transformers import util as st_util
        dense_hits = st_util.semantic_search(q, D, top_k=top_k)[0]
        dense_top = [h["corpus_id"] for h in dense_hits]

        cand_ids = list(dict.fromkeys(bm25_top + dense_top))[: 2 * top_k] or dense_top[: 2 * top_k]
        candidates = [docs[i] for i in cand_ids]
        pairs = [[query, c] for c in candidates]
        scores = _reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]

    # Otherwise, fall back to BM25 only, then naive
    idxs = _bm25_topk(query, docs, top_k) or _naive_topk(query, docs, top_k)
    return [docs[i] for i in idxs]
