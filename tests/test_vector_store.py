import numpy as np
import pytest
from rag.vector_store import VectorStore


DOCS = [
    {
        "text": "Attention mechanisms revolutionized natural language processing by allowing models to focus on relevant tokens.",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani", "Shazeer"],
        "year": 2017,
        "arxiv_id": "1706.03762",
    },
    {
        "text": "BM25 is a sparse retrieval algorithm based on term frequency and inverse document frequency.",
        "title": "Okapi BM25",
        "authors": ["Robertson", "Walker"],
        "year": 1994,
        "arxiv_id": "n/a",
    },
    {
        "text": "FAISS enables fast dense vector similarity search at billion-scale using approximate nearest neighbours.",
        "title": "Billion-Scale Similarity Search",
        "authors": ["Johnson", "Douze", "Jegou"],
        "year": 2019,
        "arxiv_id": "1702.08734",
    },
    {
        "text": "Retrieval-augmented generation combines parametric and non-parametric memory for open-domain QA.",
        "title": "RAG",
        "authors": ["Lewis"],
        "year": 2020,
        "arxiv_id": "2005.11401",
    },
]


@pytest.fixture()
def store():
    vs = VectorStore()
    vs.add_docs(DOCS)
    return vs


# ---------------------------------------------------------------------------
# Test 1 — add_docs stores metadata correctly
# ---------------------------------------------------------------------------
def test_add_docs_stores_metadata():
    vs = VectorStore()
    vs.add_docs(DOCS)

    assert len(vs.documents) == len(DOCS)

    for original, stored in zip(DOCS, vs.documents):
        assert stored["text"] == original["text"]
        assert stored["title"] == original["title"]
        assert stored["authors"] == original["authors"]
        assert stored["year"] == original["year"]
        assert stored["arxiv_id"] == original["arxiv_id"]

    # BM25 index must be populated
    assert vs._bm25 is not None
    assert len(vs._tokenized_corpus) == len(DOCS)


# ---------------------------------------------------------------------------
# Test 2 — hybrid_retrieve returns dicts that contain text and metadata
# ---------------------------------------------------------------------------
def test_hybrid_retrieve_returns_metadata(store):
    results = store.hybrid_retrieve("sparse retrieval BM25", k=2)

    assert len(results) == 2
    for doc in results:
        # Every required field must be present and non-empty
        assert "text" in doc and doc["text"]
        assert "title" in doc and doc["title"]
        assert "authors" in doc
        assert "year" in doc
        assert "arxiv_id" in doc


# ---------------------------------------------------------------------------
# Test 3 — alpha controls which retriever dominates
#
#   alpha=1.0  →  only the dense (FAISS) leg contributes to RRF scores, so
#                 hybrid_retrieve must return the same top-k ordering as
#                 the plain retrieve() method.
#
#   alpha=0.0  →  only the BM25 leg contributes, so the document that best
#                 matches the query by keyword overlap should rank first.
# ---------------------------------------------------------------------------
def test_alpha_controls_retriever_blend(store):
    query = "BM25 term frequency sparse algorithm"
    k = 3

    # --- alpha=1.0 must match pure dense ---
    hybrid_dense = store.hybrid_retrieve(query, k=k, alpha=1.0)
    pure_dense = store.retrieve(query, k=k)

    assert [d["title"] for d in hybrid_dense] == [d["title"] for d in pure_dense], (
        "With alpha=1.0, hybrid_retrieve should return the same order as retrieve()"
    )

    # --- alpha=0.0 must put the BM25-best doc first ---
    # The query is loaded with BM25-friendly keywords ("BM25", "term", "frequency",
    # "sparse", "algorithm") that all appear verbatim in the Okapi BM25 document.
    hybrid_bm25 = store.hybrid_retrieve(query, k=k, alpha=0.0)
    assert hybrid_bm25[0]["title"] == "Okapi BM25", (
        "With alpha=0.0, the document with the highest BM25 score should rank first"
    )
