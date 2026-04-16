from pypdf import PdfReader


def read_pdf(path: str) -> str:
    """Extract and return all text from a PDF file, one page at a time."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)
