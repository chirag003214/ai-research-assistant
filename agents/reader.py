from llm import call_llm
from src.pipeline import add_docs
from equations.extractor import extract_equations
from equations.latex_cleaner import clean_latex

def read_papers(papers):
    summaries = []

    for paper in papers:
        text = paper.get("summary", "")

        equations = extract_equations(text)
        equations = [clean_latex(e) for e in equations[:3]]  # limit

        prompt = f"""
Summarize the paper in 5 bullet points.

Paper content:
{text}

If equations are present, explain their physical meaning briefly.

Equations:
{equations}
"""
        summary = call_llm(prompt, max_tokens=350)

        summaries.append({
            "summary": summary,
            "equations": equations
        })
        add_docs([summary])

    return summaries




