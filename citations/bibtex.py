import os

BIB_DIR = "outputs"
os.makedirs(BIB_DIR, exist_ok=True)

def generate_bibtex(papers, filename="references.bib"):
    entries = []

    for p in papers:
        key = f"{p['authors'][0].split()[-1]}{p['year']}"
        entry = f"""
@article{{{key},
  title={{ {p['title']} }},
  author={{ {' and '.join(p['authors'])} }},
  year={{ {p['year']} }},
  eprint={{ {p['arxiv_id']} }},
  archivePrefix={{arXiv}},
  primaryClass={{quant-ph}},
}}
"""
        entries.append(entry)

    path = os.path.join(BIB_DIR, filename)
    with open(path, "w") as f:
        f.write("\n".join(entries))

    return path
