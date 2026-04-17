def clean_latex(eq: str) -> str:
    eq = eq.replace("$$", "")
    eq = eq.replace("$", "")
    return eq.strip()
