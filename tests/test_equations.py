import pytest
from equations.extractor import extract_equations


# ---------------------------------------------------------------------------
# Test 1 — inline math: $x^2$ is extracted
# ---------------------------------------------------------------------------
def test_inline_math_extracted():
    results = extract_equations(r"The function $x^2$ is convex.")
    assert r"$x^2$" in results


# ---------------------------------------------------------------------------
# Test 2 — display math: $$E=mc^2$$ is extracted intact
# ---------------------------------------------------------------------------
def test_display_math_extracted():
    results = extract_equations(r"Einstein showed $$E=mc^2$$ in 1905.")
    assert r"$$E=mc^2$$" in results


# ---------------------------------------------------------------------------
# Test 3 — display math is NOT split by inline pattern.
#           $$x + y$$ must appear as one token, not as "$$" + "$$".
#           Before the combined-regex fix the loop approach would emit
#           two empty inline matches ("$$" at each end) alongside the
#           correct display match.
# ---------------------------------------------------------------------------
def test_display_math_not_split_by_inline():
    results = extract_equations(r"$$x + y$$")

    assert r"$$x + y$$" in results
    # The inline pattern must not have matched the bare delimiter pairs
    assert "$$" not in [r for r in results if r == "$$"]


# ---------------------------------------------------------------------------
# Test 4 — no equations returns empty list
# ---------------------------------------------------------------------------
def test_no_equations_returns_empty():
    results = extract_equations("This text has no math at all.")
    assert results == []
