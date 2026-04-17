# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Multi-agent AI pipeline for academic literature review. Input: research topic → Output: structured review, gap analysis, BibTeX export. Deployed on Render.com.

## Running the App

```bash
# Activate venv
source venv/Scripts/activate        # Git Bash on Windows

pip install -r requirements.txt

streamlit run app.py                # UI at http://localhost:8501
streamlit run app.py -- --no-eval  # Skip RAGAS eval (faster, no OpenAI cost)
python main.py                      # CLI entry point (hardcoded topic)
```

Required env vars in `.env`:
- `GROQ_API_KEY` — LLM inference (required)
- `OPENAI_API_KEY` — RAGAS evaluation judge (`gpt-3.5-turbo`); without it, all RAGAS scores return `0.0`
- `COHERE_API_KEY` — optional; enables Cohere reranker. Without it, the `reranker` strategy falls back to `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Running Tests

```bash
pytest tests/                                              # all tests
pytest tests/test_hybrid_retriever.py -v                   # retrieval strategies only
pytest tests/test_vector_store.py::test_alpha_controls_retriever_blend -v  # single test
```

Tests cover FAISS+BM25 metadata and alpha blending (`test_vector_store.py`) plus smoke tests for all four retrieval strategies with no LLM or network calls (`test_hybrid_retriever.py`).

## Architecture

### Agent pipeline

```
topic → Planner → arXiv search → Reader → Critic → Synthesizer → report + BibTeX
```

Each agent in `agents/` calls `llm.call_llm()` which MD5-hashes the prompt and reads from `cache/<hash>.json` before hitting Groq — repeated runs with the same prompt cost zero API calls. Delete `cache/` to force fresh LLM responses.

### Two retrieval systems (important)

There are two separate vector stores and they are **not the same index**:

| Module | Used by | Backend |
|---|---|---|
| `rag/vector_store.py` (`store` singleton) | `agents/reader.py`, `agents/critic.py`, `agents/synthesizer.py` | FAISS + BM25, stores rich metadata dicts `{text, title, authors, year, arxiv_id}` |
| `src/pipeline.py` (`default_retriever`) | Tab 2 benchmark mode only | `HybridRetriever` singleton, stores plain strings |

`app.py` calls `store.reset()` before each pipeline run to clear stale data. The `src/pipeline.HybridRetriever` is **only used in the benchmark tab** — the main pipeline agents all go through `rag/vector_store.store`.

> **Note:** The existing CLAUDE.md previously said critic/synthesizer used `src/pipeline.py` — that is incorrect. Both call `store.hybrid_retrieve()` on the `rag.vector_store.store` singleton.

### rag/vector_store.py — Primary agent index

```python
store = VectorStore()  # module-level singleton shared by all agents
```

Key methods:
- `add_docs(docs: list[dict])` — expects dicts with `{text, title, authors, year, arxiv_id}`; embeds text with `all-MiniLM-L6-v2`, adds to FAISS, rebuilds BM25
- `retrieve(query, k=3)` → pure dense FAISS retrieval, returns full dicts
- `hybrid_retrieve(query, k=3, alpha=0.5)` → RRF fusion; `alpha=1.0` is pure dense, `alpha=0.0` is pure BM25
- `reset()` — clears all state; called by `app.py` before each run

### Retrieval layer (`src/retrieval/`) — Benchmark tab only

`HybridRetriever` composes three backends operating on plain strings:

- **Dense** — FAISS + `all-MiniLM-L6-v2` (384-dim, L2 distance)
- **Sparse** — BM25Okapi (rebuilds full index on every `add_docs` call)
- **Hybrid** — Reciprocal Rank Fusion (RRF, constant `k=60`) over dense + sparse
- **Reranker** — RRF first, then Cohere API or `cross-encoder/ms-marco-MiniLM-L-6-v2` second pass

Switch strategy at runtime: `retriever.strategy = "sparse"`.

### Citation grounding (synthesizer.py)

Synthesizer prefixes each retrieved chunk with its citation key (`[AuthorYear]`) and instructs the LLM to cite every factual claim with `(AuthorYear)`. This tethers the review to retrieved evidence and reduces hallucination.

`_citation_key(chunk)` extracts `LastName + Year` from chunk metadata.

### Benchmark mode (app.py Tab 2)

Fetches 20 arXiv papers for the query, then runs all four strategies in parallel via `ThreadPoolExecutor(max_workers=4)`. Each strategy retrieves top-5, measures latency, creates a dummy answer, and optionally evaluates with RAGAS. Results are logged to `eval_results/eval_logs.jsonl` (append-only JSONL) and displayed as a comparison table, bar chart (latency), and radar chart (RAGAS metrics).

Pass `--no-eval` flag to default the "Skip RAGAS" checkbox to checked.

### LLM gateway (`llm.py`)

Single function `call_llm(prompt, max_tokens, retries, wait_time)`. Model constant is `MODEL = "groq/llama-3.1-8b-instant"` at the top of the file — change it there to swap providers (LiteLLM routing supports any provider string). On `RateLimitError`, retries with exponential backoff + jitter: `wait_time * (2 ** attempt) + random(0, 2)` seconds.

### arXiv search (`tools/arxiv_search.py`)

Returns `[{title, summary, url, authors, year, arxiv_id, category}, ...]`. Uses 3-second delay and 5 retries for rate limiting. `tools/pdf_reader.py` exists for full-text extraction but is not currently wired into the pipeline.

## Key Outputs

- `outputs/references.bib` — BibTeX export (deduped citation keys: `Smith2023a`, `Smith2023b`)
- `cache/<md5>.json` — LLM response cache
- `eval_results/eval_logs.jsonl` — RAGAS benchmark logs (append-only JSONL)
