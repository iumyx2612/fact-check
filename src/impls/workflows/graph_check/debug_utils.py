"""Debug-only helpers for GraphCheck workflows (retrieval previews, structured logs)."""

from __future__ import annotations

import logging
from typing import Any

DEFAULT_PREVIEW_LEN = 100


def preview_text(text: str | None, n: int = DEFAULT_PREVIEW_LEN) -> str:
    """Single-line preview for console / log inspection."""
    if not text:
        return ""
    s = text.replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[:n] + "…"


def log_retrieved_nodes(
    logger: logging.Logger,
    tag: str,
    nodes: list[Any],
    *,
    preview_len: int = DEFAULT_PREVIEW_LEN,
) -> None:
    """Log rank, score, and text preview for each retrieved node (DEBUG)."""
    logger.debug("%s: %s document(s)", tag, len(nodes))
    for i, node in enumerate(nodes):
        text = getattr(node, "text", "") or ""
        score = getattr(node, "score", None)
        prev = preview_text(text, preview_len)
        logger.debug("  [%s] score=%s preview=%s", i, score, prev)
