def clean_latex(eq):
    eq = eq.replace("$$", "")
    eq = eq.replace("$", "")
    return eq.strip()
