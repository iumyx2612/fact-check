import re

WIKI_LINK_PATTERN = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")


def wiki_to_plain_text(text: str) -> str:
    """Convert Wikipedia markup links to plain text.
    [[target|display]] -> display, [[target]] -> target.
    """
    if not text:
        return text
    return WIKI_LINK_PATTERN.sub(r"\1", str(text)).strip()


def normalize_feverous_label(label: object) -> str | None:
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

