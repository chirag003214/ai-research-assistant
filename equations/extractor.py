import re

# Common LaTeX math patterns
EQUATION_PATTERNS = [
    r"\$.*?\$",          # inline math
    r"\$\$.*?\$\$",      # display math
    r"\\\[.*?\\\]",      # \[ \]
    r"\\begin{equation}.*?\\end{equation}"
]

def extract_equations(text):
    equations = []
    for pattern in EQUATION_PATTERNS:
        matches = re.findall(pattern, text, re.DOTALL)
        equations.extend(matches)
    return list(set(equations))
