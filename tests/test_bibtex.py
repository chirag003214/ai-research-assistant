import os
import pytest
from citations.bibtex import generate_bibtex


def _read_bib(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Test 1 — normal generation: 2 distinct papers produce correct keys
# ---------------------------------------------------------------------------
def test_normal_generation(tmp_path, monkeypatch):
    monkeypatch.setattr("citations.bibtex.BIB_DIR", str(tmp_path))

    papers = [
        {"title": "Attention Is All You Need", "authors": ["Ashish Vaswani"],
         "year": 2017, "arxiv_id": "1706.03762", "category": "cs.CL"},
        {"title": "BERT", "authors": ["Jacob Devlin"],
         "year": 2018, "arxiv_id": "1810.04805", "category": "cs.CL"},
    ]
    path = generate_bibtex(papers, filename="test_normal.bib")
    content = _read_bib(path)

    assert "@article{Vaswani2017" in content
    assert "@article{Devlin2018" in content
    assert "Attention Is All You Need" in content
    assert "BERT" in content


# ---------------------------------------------------------------------------
# Test 2 — duplicate-key disambiguation: two papers by Smith in 2023
#           must produce Smith2023a and Smith2023b, not two Smith2023
# ---------------------------------------------------------------------------
def test_duplicate_key_disambiguation(tmp_path, monkeypatch):
    monkeypatch.setattr("citations.bibtex.BIB_DIR", str(tmp_path))

    papers = [
        {"title": "Paper One", "authors": ["Alice Smith"],
         "year": 2023, "arxiv_id": "2301.00001", "category": "cs.AI"},
        {"title": "Paper Two", "authors": ["Bob Smith"],
         "year": 2023, "arxiv_id": "2301.00002", "category": "cs.AI"},
    ]
    path = generate_bibtex(papers, filename="test_dedup.bib")
    content = _read_bib(path)

    assert "@article{Smith2023a" in content
    assert "@article{Smith2023b" in content
    assert "@article{Smith2023," not in content  # bare key must not appear


# ---------------------------------------------------------------------------
# Test 3 — empty authors guard: missing authors list defaults to "Unknown"
#           and must not raise IndexError
# ---------------------------------------------------------------------------
def test_empty_authors_guard(tmp_path, monkeypatch):
    monkeypatch.setattr("citations.bibtex.BIB_DIR", str(tmp_path))

    papers = [
        {"title": "Authorless Paper", "authors": [],
         "year": 2021, "arxiv_id": "2101.00000", "category": "cs.AI"},
    ]
    path = generate_bibtex(papers, filename="test_empty_authors.bib")
    content = _read_bib(path)

    assert "@article{Unknown2021" in content
