import re

# Single alternation regex: $$ must appear before $ so the engine
# commits to display math before trying inline, preventing $$ from
# being split into two empty inline matches.
_EQUATION_RE = re.compile(
    r"\$\$.*?\$\$"
    r"|\$.*?\$"
    r"|\\\[.*?\\\]"
    r"|\\begin\{equation\}.*?\\end\{equation\}",
    re.DOTALL,
)

def extract_equations(text: str) -> list[str]:
    return list(set(_EQUATION_RE.findall(text)))
