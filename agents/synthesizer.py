from llm import call_llm
from rag.vector_store import retrieve

def synthesize(topic, papers):
    context = retrieve(topic, k=5)

    citation_map = {
        p["title"]: f"{p['authors'][0].split()[-1]}{p['year']}"
        for p in papers
    }

    prompt = f"""
Write a structured literature review on:
{topic}

Use ONLY the context below.
Every factual claim MUST end with a citation key in parentheses.

Context:
{context}

Citation keys:
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


