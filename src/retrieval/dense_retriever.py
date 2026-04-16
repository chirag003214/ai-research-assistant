from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[str] = []

    def add_docs(self, texts: List[str]) -> None:
        if not texts:
            return
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings))
        self.documents.extend(texts)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not self.documents:
            return []
        q_emb = self.model.encode([query])
        # Only search up to the number of documents we have
        actual_k = min(k, len(self.documents))
        _, ids = self.index.search(np.array(q_emb), actual_k)
        return [self.documents[i] for i in ids[0] if i < len(self.documents)]
