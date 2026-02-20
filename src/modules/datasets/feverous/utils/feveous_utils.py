import re

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
