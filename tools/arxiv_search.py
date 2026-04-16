import arxiv

def search_arxiv(query, max_results=5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    # Use a custom client with heavier rate-limit fallback
    client = arxiv.Client(
        page_size=max_results, 
        delay_seconds=3.0, 
        num_retries=5
    )

    papers = []
    try:
        for r in client.results(search):
            papers.append({
                "title": r.title,
                "summary": r.summary,
                "url": r.entry_id,
                "authors": [a.name for a in r.authors],
                "year": r.published.year,
                "arxiv_id": r.entry_id.split("/")[-1],
                "category": r.primary_category
            })
    except Exception as e:
        print(f"arXiv search warning: {e}")
        
    return papers


