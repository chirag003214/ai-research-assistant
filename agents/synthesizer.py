from llm import call_llm
from rag.vector_store import store


def _citation_key(chunk: dict) -> str:
    authors = chunk.get("authors", [])
    first_author = authors[0] if authors else "Unknown"
    last_name = first_author.split()[-1] if first_author else "Unknown"
    return f"{last_name}{chunk.get('year', '')}"


def synthesize(topic, papers=None):
    chunks = store.retrieve(topic, k=5)

    if not chunks:
        context = "No documents indexed yet."
        citation_map = {}
    else:
        # Each context block is prefixed with its citation key so the LLM
        # knows exactly which key to use for each piece of evidence
        context = "\n\n".join(
            f"[{_citation_key(c)}] {c['text']}"
            for c in chunks
        )
        citation_map = {c["title"]: _citation_key(c) for c in chunks}

    prompt = f"""
Write a structured literature review on:
{topic}

Use ONLY the context below.
Every factual claim MUST end with a citation key in parentheses.

Context:
{context}

Citation keys available:
{citation_map}

Sections:
1. Background
2. Methods
3. Results
4. Limitations
5. Future Work

Format citations like: (AuthorYear)
Max 400 words.
"""
    return call_llm(prompt, max_tokens=600)
