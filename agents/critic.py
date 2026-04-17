from llm import call_llm
from rag.vector_store import store


def critique(topic: str) -> str:
    chunks = store.hybrid_retrieve(topic)

    if not chunks:
        context = "No documents indexed yet."
    else:
        context = "\n\n".join(
            f"[{c['title']} ({c['year']})]: {c['text']}"
            for c in chunks
        )

    prompt = f"""
You are a peer reviewer.

Using the following research summaries:
{context}

Identify:
- common limitations
- research gaps
- weaknesses in existing methods
(max 150 words)
"""
    return call_llm(prompt, max_tokens=300)
