# 🧠 AI Research Assistant (Physics & ML)

A citation-grounded, multi-agent AI research assistant for academic literature review.

## Features
- Multi-document synthesis
- Retrieval-Augmented Generation (RAG)
- Strict citation grounding
- BibTeX / LaTeX export
- Equation → LaTeX extraction
- Interactive Streamlit UI

## Tech Stack
- Groq LLMs (via LiteLLM)
- FAISS (vector search)
- SentenceTransformers
- Streamlit

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
