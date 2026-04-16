from typing import List, Literal
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.reranker import Reranker
from collections import defaultdict

Strategy = Literal["dense", "sparse", "hybrid", "reranker"]

class HybridRetriever:
    def __init__(self, strategy: Strategy = "hybrid"):
        self.strategy = strategy
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        
        # Only initialize reranker if we need it, though Reranker class itself is lazy somewhat.
        # But let's initialize it here for simplicity and caching of the model.
        self.reranker = Reranker()

    def add_docs(self, texts: List[str]) -> None:
        if not texts:
            return
        self.dense.add_docs(texts)
        self.sparse.add_docs(texts)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not self.dense.documents:
            return ["No documents indexed yet."]
            
        if self.strategy == "dense":
            return self.dense.retrieve(query, k=k)
        elif self.strategy == "sparse":
            return self.sparse.retrieve(query, k=k)
        
        # Determine how many items to retrieve initially for fusion/reranking
        fetch_k = 20 if self.strategy == "reranker" else k * 2
        
        dense_results = self.dense.retrieve(query, k=fetch_k)
        sparse_results = self.sparse.retrieve(query, k=fetch_k)
        
        # Reciprocal Rank Fusion
        rrf_scores = defaultdict(float)
        rank_constant = 60
        
        for rank, doc in enumerate(dense_results):
            rrf_scores[doc] += 1.0 / (rank_constant + rank + 1)
            
        for rank, doc in enumerate(sparse_results):
            rrf_scores[doc] += 1.0 / (rank_constant + rank + 1)
            
        hybrid_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_docs = [doc for doc, score in hybrid_results]
        
        if self.strategy == "reranker":
            return self.reranker.rerank(query, hybrid_docs, top_k=k)
        
        return hybrid_docs[:k]
