import pytest
from src.retrieval.hybrid_retriever import HybridRetriever

@pytest.fixture
def sample_docs():
    return [
        "Quantum computing enables exponential speedup for factoring.",
        "Generative adversarial networks use two neural networks competing against each other.",
        "Attention mechanisms revolutionized natural language processing.",
        "BM25 is a popular sparse retrieval algorithm.",
        "FAISS provides fast dense vector similarity search."
    ]

def test_dense_retrieval(sample_docs):
    retriever = HybridRetriever(strategy="dense")
    retriever.add_docs(sample_docs)
    results = retriever.retrieve("What algorithm does sparse retrieval use?", k=2)
    assert len(results) == 2
    assert type(results[0]) is str

def test_sparse_retrieval(sample_docs):
    retriever = HybridRetriever(strategy="sparse")
    retriever.add_docs(sample_docs)
    results = retriever.retrieve("neural networks competing", k=2)
    assert len(results) == 2
    assert "Generative adversarial networks" in results[0]

def test_hybrid_retrieval(sample_docs):
    retriever = HybridRetriever(strategy="hybrid")
    retriever.add_docs(sample_docs)
    results = retriever.retrieve("fast dense search FAISS", k=3)
    assert len(results) == 3
    
    # Check if fusion produced reasonable outcome
    assert any("FAISS" in doc for doc in results)

def test_reranker_fallback(sample_docs):
    retriever = HybridRetriever(strategy="reranker")
    retriever.add_docs(sample_docs)
    results = retriever.retrieve("Attention mechanisms", k=1)
    # The CrossEncoder/fallback should run and return exactly k items
    assert len(results) == 1
    assert "Attention" in results[0]
