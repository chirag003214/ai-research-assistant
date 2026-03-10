# 🧠 AI Research Assistant

A multi-agent AI pipeline that automates academic literature review.
Enter a research topic → get a structured review, gap analysis, and BibTeX export.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## How it works
```
Topic → Planner → arXiv Search → Reader (RAG) → Critic → Synthesizer → Report + BibTeX
```

Four specialized agents collaborate via a shared FAISS vector store:

| Agent | Role |
|---|---|
| **Planner** | Decomposes topic into focused research questions |
| **Reader** | Summarizes papers, extracts equations, indexes into vector store |
| **Critic** | Identifies gaps and limitations using RAG over stored summaries |
| **Synthesizer** | Writes citation-grounded literature review |

## Tech Stack

- **LLM:** Groq `llama-3.1-8b-instant` via LiteLLM (swappable to GPT-4/Claude in one line)
- **Vector search:** FAISS + `all-MiniLM-L6-v2` embeddings (384-dim)
- **Paper source:** arXiv API
- **UI:** Streamlit
- **Deployment:** Render.com

## Quick Start
```bash
git clone https://github.com/chirag003214/ai-research-assistant
cd ai-research-assistant

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY to .env

streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

## Project Structure
```
agents/         — Planner, Reader, Critic, Synthesizer
rag/            — FAISS vector store + SentenceTransformer embeddings
tools/          — arXiv search, PDF reader
equations/      — LaTeX equation extraction and cleaning
citations/      — BibTeX export
llm.py          — LLM gateway with prompt caching and retry logic
app.py          — Streamlit UI
```

## Key Design Decisions

- **Prompt caching:** LLM responses cached by MD5 hash to avoid redundant API calls
- **RAG grounding:** Critic and Synthesizer retrieve from vector store, not raw LLM memory
- **LiteLLM abstraction:** Swap to any LLM provider by changing one constant in `llm.py`
- **Citation grounding:** Synthesizer builds a citation map and instructs the LLM to use real keys

## Limitations / Future Work

- Currently uses arXiv abstracts only; full PDF parsing is the next step (`pypdf` is in requirements)
- Vector store is in-memory (reset on restart); would replace with ChromaDB for persistence
- Pipeline runs sequentially; parallelizing the reader with `ThreadPoolExecutor` would cut latency ~60%
