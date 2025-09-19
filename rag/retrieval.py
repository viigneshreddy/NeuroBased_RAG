# rag/retrieval.py
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# Try BM25, fall back gracefully if missing
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Strong reranker (bigger); if it’s heavy/slow, switch to the smaller MiniLM line below.
_RERANK_MODEL = "BAAI/bge-reranker-base"
# _RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # <- smaller fallback

_embedder = SentenceTransformer(_EMBED_MODEL)
_reranker = CrossEncoder(_RERANK_MODEL)

def hybrid_retrieve(query: str, docs: list[str], top_k: int = 3) -> list[str]:
    if not docs:
        return []

    # 1) BM25 lexical (if available)
    bm25_top = []
    if HAS_BM25:
        tokenized = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_top = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    # 2) Dense semantic
    D = _embedder.encode(docs, convert_to_tensor=True, normalize_embeddings=True)
    q = _embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    dense_hits = util.semantic_search(q, D, top_k=top_k)[0]
    dense_top = [h["corpus_id"] for h in dense_hits]

    # 3) Merge candidates
    cand_ids = list(dict.fromkeys(bm25_top + dense_top))[: 2 * top_k] if HAS_BM25 else dense_top[: 2 * top_k]
    candidates = [docs[i] for i in cand_ids]

    # 4) Rerank with CrossEncoder
    pairs = [[query, c] for c in candidates]
    scores = _reranker.predict(pairs)  # << correct API
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
