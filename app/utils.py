import re
from bs4 import BeautifulSoup


def clean_html_to_text(html: str) -> str:
    """Convert raw HTML to cleaner plain text."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noisy tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n +", "\n", text)

    # Remove very short junk lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) > 2]

    return "\n".join(lines)