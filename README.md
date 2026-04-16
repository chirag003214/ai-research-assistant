# AI Research Assistant

An AI pipeline that automates academic literature review. Enter a research topic and get a structured review, gap analysis, and BibTeX export — all grounded in real arXiv papers.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## How It Works

```
Topic → Planner → arXiv Search → Reader → Critic → Synthesizer → Report + BibTeX
```

Four sequential stages collaborate through a shared FAISS vector store:

| Stage | What it does |
|---|---|
| **Planner** | Decomposes the topic into 4 focused research questions |
| **Reader** | Summarizes each paper, extracts LaTeX equations, indexes summaries into FAISS |
| **Critic** | Retrieves indexed summaries and identifies gaps, limitations, and weak methods |
| **Synthesizer** | Writes a 5-section literature review with inline citation keys, grounded in retrieved context |

## Tech Stack

- **LLM:** Groq `llama-3.1-8b-instant` via LiteLLM (swap to any provider by changing one constant in `llm.py`)
- **Vector search:** FAISS + `all-MiniLM-L6-v2` embeddings (384-dim, in-memory)
- **Paper source:** arXiv API (abstract-level; full PDF extraction available via `pypdf`)
- **Prompt caching:** LLM responses cached by MD5 hash in `cache/` — repeated runs cost zero API calls
- **UI:** Streamlit with a Research Pipeline tab and a Retrieval Benchmark tab
- **Deployment:** Render.com (`render.yaml` included)

## Quick Start

```bash
git clone https://github.com/chirag003214/ai-research-assistant
cd ai-research-assistant

pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY to .env

streamlit run app.py       # UI at http://localhost:8501
python main.py             # CLI mode (hardcoded topic)
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Retrieval Benchmark Tab

The second tab benchmarks four retrieval strategies against a live arXiv corpus:

| Strategy | Method |
|---|---|
| Dense | FAISS cosine similarity with MiniLM embeddings |
| Sparse | BM25 term-frequency scoring |
| Hybrid | Reciprocal Rank Fusion (RRF) over dense + sparse |
| Reranker | RRF followed by Cohere API or CrossEncoder second pass |

Strategies run in parallel via `ThreadPoolExecutor`. Results are displayed as a comparison table, latency bar chart, and RAGAS metrics radar chart. Evaluation requires `OPENAI_API_KEY`; pass `--no-eval` to skip it.

Benchmark logs append to `eval_results/eval_logs.jsonl`.

## Key Design Decisions

- **Prompt caching:** `call_llm()` in `llm.py` MD5-hashes each prompt and reads from disk before hitting the API. Delete `cache/` to force fresh responses.
- **Citation grounding:** Synthesizer builds a citation map (`AuthorYear` keys) and instructs the LLM to cite every factual claim — reducing hallucination in the review.
- **LiteLLM abstraction:** One constant (`MODEL` in `llm.py`) controls the provider. Swap to GPT-4, Claude, or any other LiteLLM-supported model without touching agent code.
- **PDF extraction:** `tools/pdf_reader.py` provides `read_pdf(path) -> str` via pypdf, ready for the Reader agent to use with full-text PDFs.

## Outputs

| File | Contents |
|---|---|
| `citations/<topic>.bib` | BibTeX entries for all retrieved papers |
| `cache/<md5>.json` | Cached LLM responses |
| `eval_results/eval_logs.jsonl` | Benchmark evaluation logs |

## Future Work

- Wire `pdf_reader` into the Reader agent for full-text extraction instead of abstracts only
- Persist the vector store with ChromaDB (currently in-memory, reset on restart)
- Parallelize the Reader stage with `ThreadPoolExecutor` to cut latency ~60%
- Add self-correction loops (CRAG-style) where the Critic can trigger re-retrieval
- Replace the RAGAS benchmark's dummy answers with real LLM-generated responses
