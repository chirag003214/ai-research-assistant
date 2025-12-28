import arxiv

def search_arxiv(query, max_results=5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for r in search.results():
        papers.append({
            "title": r.title,
            "summary": r.summary,
            "url": r.entry_id,
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "arxiv_id": r.entry_id.split("/")[-1]
        })
    return papers


