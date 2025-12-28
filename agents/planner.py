from llm import call_llm

def plan_research(topic):
    prompt = f"""
Break the research topic into 4 focused research questions.
Topic: {topic}
Return as bullet points.
"""
    return call_llm(prompt, max_tokens=200)

