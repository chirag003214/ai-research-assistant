import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.documents: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def add_docs(self, docs: list[dict]) -> None:
        """
        Index documents with full metadata.

        Each dict must contain:
            text     (str)   — the content to embed
            title    (str)
            authors  (list)
            year     (int|str)
            arxiv_id (str)
        """
        if not docs:
            return
        texts = [d["text"] for d in docs]
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(docs)

        # Rebuild BM25 over the full corpus each time docs are added.
        # Tokenization: lowercase + whitespace split (simple, fast, no stopwords).
        self._tokenized_corpus = [doc["text"].lower().split() for doc in self.documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return up to k document dicts ranked by embedding similarity."""
        if not self.documents:
            return []
        q_emb = self.model.encode([query])
        actual_k = min(k, len(self.documents))
        _, ids = self.index.search(np.array(q_emb, dtype=np.float32), actual_k)
        return [self.documents[i] for i in ids[0] if i < len(self.documents)]

    def hybrid_retrieve(self, query: str, k: int = 3, alpha: float = 0.5) -> list[dict]:
        """
        Hybrid dense + sparse retrieval via Reciprocal Rank Fusion (RRF).

        How RRF works
        -------------
        Each retrieval system (FAISS, BM25) independently ranks all candidate
        documents. Instead of combining raw scores (which live on incomparable
        scales), we use each document's *rank position*:

            RRF_score(doc) = alpha       * 1 / (dense_rank + 60)
                           + (1 - alpha) * 1 / (bm25_rank  + 60)

        The constant 60 is the standard smoothing value from Cormack et al.
        (SIGIR 2009). It flattens the contribution curve so the difference
        between rank-1 and rank-2 is not disproportionately large, making the
        fusion robust when one system ranks a document very highly by accident.

        alpha controls the blend:
            alpha = 1.0  →  pure dense  (FAISS embedding similarity only)
            alpha = 0.0  →  pure sparse (BM25 keyword matching only)
            alpha = 0.5  →  equal weight (default)

        Steps
        -----
        1. FAISS dense retrieval  → top-2k candidates by cosine/L2 distance
        2. BM25 sparse retrieval  → top-2k candidates by TF-IDF-like score
        3. RRF score accumulation over both ranked lists
        4. Sort by combined RRF score, return top-k dicts with text + metadata
        """
        if not self.documents or self._bm25 is None:
            return []

        fetch_k = min(2 * k, len(self.documents))

        # --- Dense leg (FAISS) ---
        q_emb = self.model.encode([query])
        _, dense_ids = self.index.search(np.array(q_emb, dtype=np.float32), fetch_k)
        dense_ranked: list[int] = [i for i in dense_ids[0] if i < len(self.documents)]

        # --- Sparse leg (BM25) ---
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_ranked: list[int] = list(np.argsort(bm25_scores)[::-1][:fetch_k])

        # --- Reciprocal Rank Fusion ---
        rrf: dict[int, float] = {}

        for rank, doc_idx in enumerate(dense_ranked):
            rrf[doc_idx] = rrf.get(doc_idx, 0.0) + alpha / (rank + 60)

        for rank, doc_idx in enumerate(bm25_ranked):
            rrf[doc_idx] = rrf.get(doc_idx, 0.0) + (1.0 - alpha) / (rank + 60)

        top_ids = sorted(rrf, key=lambda idx: rrf[idx], reverse=True)[:k]
        return [self.documents[i] for i in top_ids]

    def reset(self) -> None:
        """Clear the index and stored documents — call before each pipeline run."""
        self.index = faiss.IndexFlatL2(384)
        self.documents = []
        self._bm25 = None
        self._tokenized_corpus = []


# Shared instance used by all agents
store = VectorStore()
