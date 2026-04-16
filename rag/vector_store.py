import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.documents: list[dict] = []

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

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return up to k document dicts ranked by embedding similarity."""
        if not self.documents:
            return []
        q_emb = self.model.encode([query])
        actual_k = min(k, len(self.documents))
        _, ids = self.index.search(np.array(q_emb, dtype=np.float32), actual_k)
        return [self.documents[i] for i in ids[0] if i < len(self.documents)]

    def reset(self) -> None:
        """Clear the index and stored documents — call before each pipeline run."""
        self.index = faiss.IndexFlatL2(384)
        self.documents = []


# Shared instance used by all agents
store = VectorStore()
