import os
from typing import List
from dataclasses import dataclass

@dataclass
class RankedDocument:
    document: str
    score: float

class Reranker:
    def __init__(self):
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.cohere_client = None
        self.cross_encoder = None
        
        if self.cohere_api_key:
            try:
                import cohere
                self.cohere_client = cohere.Client(self.cohere_api_key)
            except ImportError:
                print("Cohere library not found. Falling back to CrossEncoder.")
                pass
        
        if not self.cohere_client:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                print(f"Failed to load CrossEncoder fallback: {e}")

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        if not documents:
            return []
            
        if self.cohere_client:
            try:
                # Cohere expects docs to be short. Max limits apply.
                results = self.cohere_client.rerank(
                    query=query, 
                    documents=documents, 
                    top_n=top_k, 
                    model='rerank-english-v3.0'
                )
                return [documents[r.index] for r in results.results]
            except Exception as e:
                print(f"Cohere rerank failed, falling back to basic top_k: {e}")
        
        if self.cross_encoder:
            try:
                pairs = [[query, doc] for doc in documents]
                scores = self.cross_encoder.predict(pairs)
                # Ensure scores.tolist() or list of floats
                scores = scores.tolist() if hasattr(scores, 'tolist') else scores
                ranked_pairs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
                return [doc for score, doc in ranked_pairs[:top_k]]
            except Exception as e:
                print(f"CrossEncoder rerank failed: {e}")
                
        # Graceful fallback: just return the top_k without reranking
        return documents[:top_k]
