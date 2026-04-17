import os

BIB_DIR = "outputs"
os.makedirs(BIB_DIR, exist_ok=True)

def generate_bibtex(papers: list[dict], filename: str = "references.bib") -> str:
    # --- pass 1: build raw keys ---
    raw_keys = []
    for p in papers:
        first_author = p['authors'][0] if p['authors'] else "Unknown"
        raw_keys.append(f"{first_author.split()[-1]}{p['year']}")

    # --- disambiguate duplicates: Smith2023 → Smith2023a, Smith2023b … ---
    from collections import Counter
    key_counts = Counter(raw_keys)
    key_seen   = {}
    final_keys = []
    for k in raw_keys:
        if key_counts[k] > 1:
            idx = key_seen.get(k, 0)
            final_keys.append(f"{k}{chr(ord('a') + idx)}")
            key_seen[k] = idx + 1
        else:
            final_keys.append(k)

    # --- pass 2: render entries with disambiguated keys ---
    entries = []
    for key, p in zip(final_keys, papers):
        entry = f"""
@article{{{key},
  title={{ {p['title']} }},
  author={{ {' and '.join(p['authors'])} }},
  year={{ {p['year']} }},
  eprint={{ {p['arxiv_id']} }},
  archivePrefix={{arXiv}},
  primaryClass={{ {p.get('category', 'cs.AI')} }},
}}
"""
        entries.append(entry)

    path = os.path.join(BIB_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))

    return path
