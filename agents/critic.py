from llm import call_llm
from rag.vector_store import retrieve

def critique(topic):
    context = retrieve(topic)

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

