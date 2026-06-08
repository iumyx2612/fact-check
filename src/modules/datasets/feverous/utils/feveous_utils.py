import re

WIKI_LINK_PATTERN = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")


def normalize_feverous_label(label: object) -> str | None:
    """
    Normalize FEVEROUS labels to a standard format of other datasets
    """
    if label is None:
        return None
    if not isinstance(label, str):
        label = str(label)
    norm = label.strip().upper()
    if norm == "SUPPORTS":
        return "SUPPORT"
    if norm == "REFUTES":
        return "REFUTE"
    if norm in {"NOT ENOUGH INFO", "NOT_ENOUGH_INFO"}:
        return "NEI"
    return norm or None


def wiki_to_plain_text(text: str) -> str:
    """Convert Wikipedia markup links to plain text.
    [[target|display]] -> display, [[target]] -> target.
    """
    if not text:
        return text

    text = re.sub(r'\[H\]\s*', '', text)

    return WIKI_LINK_PATTERN.sub(r"\1", str(text)).strip()


def wiki_links_to_md_links(text):
    return re.sub(r'\[\[([^|]+)\|([^]]+)]]', r'[\2](\1)', text)


def wiki_table_to_md(text: str) -> str:
    lines = [wiki_to_plain_text(line) for line in text.strip().split("\n")]

    # Split header and rows
    header = lines[0].split(" | ")
    rows = [line.split(" | ") for line in lines[1:]]

    # Build Markdown table
    markdown = []
    markdown.append("| " + " | ".join(header) + " |")
    markdown.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in rows:
        markdown.append("| " + " | ".join(row) + " |")

    markdown = "\n".join(markdown)
    return markdown