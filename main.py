from agents.planner import plan_research
from agents.reader import read_papers
from agents.critic import critique
from agents.synthesizer import synthesize
from tools.arxiv_search import search_arxiv
from citations.bibtex import generate_bibtex

bib_path = generate_bibtex(papers)
print(f"BibTeX saved to {bib_path}")

topic = "Quantum Machine Learning for Classification"

print("\n--- Planning ---")
plan = plan_research(topic)
print(plan)

print("\n--- Searching Papers ---")
papers = search_arxiv(topic)

print("\n--- Reading Papers ---")
summaries = read_papers(papers)

print("\n--- Critiquing Research ---")
crit = critique(topic)
print(crit)

print("\n--- Synthesizing Report ---")
final_report = synthesize(topic)
print(final_report)

