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
python main.py                      # CLI entry point (hardcoded topic)
```

Required env vars in `.env`:
- `GROQ_API_KEY` — LLM inference (required)
- `OPENAI_API_KEY` — RAGAS evaluation judge (`gpt-3.5-turbo`); without it, all RAGAS scores return `0.0`
- `COHERE_API_KEY` — optional; enables Cohere reranker. Without it, the `reranker` strategy falls back to `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Running Tests

```bash
pytest tests/                        # all tests
pytest tests/test_hybrid_retriever.py -v    # single file
```

Tests cover all four retrieval strategies (`dense`, `sparse`, `hybrid`, `reranker`) via `HybridRetriever` directly — no LLM or network calls.

## Architecture

### Agent pipeline

```
topic → Planner → arXiv search → Reader → Critic → Synthesizer → report + BibTeX
```

Each agent in `agents/` calls `llm.call_llm()` which MD5-hashes the prompt and reads from `cache/<hash>.json` before hitting Groq — repeated runs with the same prompt cost zero API calls. Delete `cache/` to force fresh LLM responses.

### Two retrieval systems (important)

There are two separate vector stores in this codebase and they are **not the same index**:

| Module | Used by | Backend |
|---|---|---|
| `rag/vector_store.py` | `agents/critic.py` | Module-level FAISS globals (`index`, `documents`) |
| `src/pipeline.py` | `agents/reader.py`, `agents/synthesizer.py` | `HybridRetriever` singleton (`default_retriever`) |

`reader.py` calls `src/pipeline.add_docs()` → populates `HybridRetriever`. But `critic.py` calls `rag.vector_store.retrieve()` → queries the old FAISS globals. These are never synchronized, so **Critic retrieves from a permanently empty index** unless `rag/vector_store.add_docs()` is also called somewhere. This is a known inconsistency.

### Retrieval layer (`src/retrieval/`)

`HybridRetriever` composes three backends:

- **Dense** — FAISS + `all-MiniLM-L6-v2` (384-dim)
- **Sparse** — BM25Okapi (rebuilds full BM25 index on every `add_docs` call — expensive for large corpora)
- **Hybrid** — Reciprocal Rank Fusion (RRF, `k=60`) over dense + sparse results
- **Reranker** — RRF first, then Cohere API or CrossEncoder second pass

Switching strategy is a single attribute: `retriever.strategy = "dense"`.

### Benchmark mode (app.py Tab 2)

Runs all four strategies in parallel via `ThreadPoolExecutor(max_workers=4)`, evaluates with RAGAS (skippable with the checkbox or `--no-eval` CLI flag), logs results to `eval_results/eval_logs.jsonl`. Pass `--no-eval` to `streamlit run app.py` to default the checkbox to checked.

### LLM gateway (`llm.py`)

Single function `call_llm(prompt, max_tokens, retries, wait_time)`. Model constant is `MODEL = "groq/llama-3.1-8b-instant"` at the top of the file — change it there to swap providers (LiteLLM routing means any supported provider string works).

## Key Outputs

- `outputs/` — synthesized reports
- `citations/<topic>.bib` — BibTeX export
- `cache/<md5>.json` — LLM response cache
- `eval_results/eval_logs.jsonl` — RAGAS benchmark logs (append-only JSONL)
