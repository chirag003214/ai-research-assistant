from typing import List
from rank_bm25 import BM25Okapi
import re

class SparseRetriever:
    def __init__(self):
        self.documents: List[str] = []
        self.bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def add_docs(self, texts: List[str]) -> None:
        if not texts:
            return
        self.documents.extend(texts)
        tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not self.bm25 or not self.documents:
            return ["No documents indexed yet."]
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Get top k indices
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_k_indices]
