from collections import defaultdict
from typing import Optional

from .utils.annotation_processor import AnnotationProcessor
from .utils.wiki_page import WikiPage
from .database.feverous_db import FeverousDB
from ..base import Dataset
from ...utils import normalize_feverous_label, wiki_to_plain_text

class Feverous(Dataset):
    def __init__(self,
                 annotations: AnnotationProcessor,
                 wiki_db: Optional[FeverousDB] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.annotations = annotations
        self.wiki_db = wiki_db

    @classmethod
    def from_path(cls, dataset_path: str, db_path: Optional[str] = None):
        anno_processor = AnnotationProcessor(dataset_path)
        wiki_db = FeverousDB(db_path)

        return cls(anno_processor, wiki_db, claims=None)

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        for annotation in self.annotations:
            claim = annotation.get_claim()

            try:
                challenge = annotation.get_challenge()
                label = normalize_feverous_label(annotation.get_verdict())
                context_dicts = annotation.get_context(flat=True)
                evidences = annotation.get_evidence(flat=True)

                evidence_str = ""
                context_str = ""
                # Process evidence + context
                for i, evidence in enumerate(evidences):
                    wiki_doc = evidence.split('_')[0]
                    evidence_id = '_'.join(evidence.split('_')[1:])

                    page_json = self.wiki_db.get_doc_json(wiki_doc)
                    wiki_page = WikiPage(wiki_doc, page_json)

                    # sentence: handled implicitly via get_element_by_id (sentence in page_items)
                    # title: explicit handling needed (title not in page_items)
                    # cell/header_cell, item, table_caption: need specialized getters
                    content = wiki_page.get_element_by_id(evidence_id)
                    if "title" in evidence_id:
                        content = "Title: " + wiki_page.get_title_content()
                    elif "cell" in evidence_id:
                        content = "Cell: " + wiki_page.get_cell_content(evidence_id)
                    elif "item" in evidence_id:
                        content = "Item: " + wiki_page.get_item_by_id(evidence_id)
                    elif "table_caption" in evidence_id:
                        content = "Table caption: " + str(
                            wiki_page.get_caption_content(evidence_id) or ""
                        )
                    evidence_str += f"- Evidence {i+1}: {str(content)}\n"

                for i, (evidence_context, contexts) in enumerate(context_dicts.items()):
                    for j, context in enumerate(contexts):
                        wiki_doc = context.split('_')[0]
                        context_id = '_'.join(context.split('_')[1:])

                        page_json = self.wiki_db.get_doc_json(wiki_doc)
                        wiki_page = WikiPage(wiki_doc, page_json)

                        # Same evidence-type handling as above (sentence implicit, title/cell/item/table_caption explicit)
                        content = wiki_page.get_element_by_id(context_id)
                        if "title" in context_id:
                            content = "Title: " + wiki_page.get_title_content()
                        elif "cell" in context_id:
                            content = "Cell: " + wiki_page.get_cell_content(context_id)
                        elif "item" in context_id:
                            content = "Item: " + wiki_page.get_item_by_id(context_id)
                        elif "table_caption" in context_id:
                            content = "Table caption: " + str(
                                wiki_page.get_caption_content(context_id) or ""
                            )
                        context_str += f"- Context {i+1}_{j+1}: {str(content)}\n"

                context = context_str
                evidence = evidence_str

            except:
                challenge = None
                context = None
                evidence = None

            yield {
                "context": context,
                "claim": claim,
                "evidence": evidence,
                "label": label
            }


class FeverousEvidenceFormat(Dataset):
    """Dataset that formats FEVEROUS evidence for LLM fact-checking.

    Output format:
    - Evidence: grouped by wiki page, with sections [Source: <title>], Passages,
      Table(s), List items. Wiki markup (e.g. [[link|text]]) is stripped to plain text.
    - Context: one line per context element, plain text.

    Evidence marking:
    - Actual evidence elements are wrapped in **bold** (passages, table cells,
      table captions when caption-only, list items). Only a subset of content
      may be evidence; bold helps the model focus.
    - Prompt hint: add something like "The **bold** parts are the actual evidence."
      when using this format in your prompt.
    """

    def __init__(
        self,
        annotations: AnnotationProcessor,
        wiki_db: Optional[FeverousDB] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.annotations = annotations
        self.wiki_db = wiki_db

    @classmethod
    def from_path(cls, dataset_path: str, db_path: Optional[str] = None):
        anno_processor = AnnotationProcessor(dataset_path)
        wiki_db = FeverousDB(db_path)
        return cls(anno_processor, wiki_db, claims=None)

    def __getitem__(self, item):
        raise NotImplementedError

    def _format_table_as_markdown(
        self,
        wiki_page: WikiPage,
        table,
        evidence_cell_ids: set[str] | None = None,
    ) -> str:
        """Format WikiTable as markdown table with plain text cells.

        Format:
            | A | B | C |
            | --- | --- | --- |
            | x | **y** | z |

        Rationale:
        - Markdown tables are standard and easy for LLMs to parse.
        - [H] prefix on header cells is stripped so output is clean.
        - wiki_to_plain_text removes [[link|text]] markup from cell content.
        - evidence_cell_ids: cells in this set are wrapped in **bold** so the model
          knows which cells are the actual evidence (vs context).
        """
        evidence_cell_ids = evidence_cell_ids or set()
        lines = []
        for row in table.rows:
            cells = []
            for cell in row.row:
                raw = str(cell)
                if raw.startswith("[H] "):
                    raw = raw[4:]
                content = wiki_to_plain_text(raw)
                if cell.name in evidence_cell_ids:
                    content = f"**{content}**"
                cells.append(content)
            lines.append("| " + " | ".join(cells) + " |")
        if lines:
            col_count = len(lines[0].split("|")[1:-1])
            header = "| " + " | ".join(["---"] * col_count) + " |"
            return "\n".join([lines[0], header] + lines[1:])
        return ""

    def _parse_table_id_from_cell_id(self, element_id: str) -> str:
        """Parse table_id from cell_id or header_cell_id.

        Format:
            cell_<table>_<row>_<col> -> table_<table>
            header_cell_<table>_<row>_<col> -> table_<table>

        Rationale:
        - Table index differs by position: cell has table at index 1, header_cell at 2.
        - Used for deduplication (one table per evidence set) and grouping.
        """
        parts = element_id.split("_")
        if element_id.startswith("cell_"):
            return "table_" + parts[1] if len(parts) > 1 else "table_?"
        if element_id.startswith("header_cell_"):
            return "table_" + parts[2] if len(parts) > 2 else "table_?"
        return "table_?"

    def _parse_table_id_from_caption_id(self, element_id: str) -> str:
        """Parse table_id from table_caption_id.

        Format:
            table_caption_<table> -> table_<table>

        Rationale:
        - Caption evidence references a table by its caption; we need table_id
          to fetch the full table and avoid duplicate displays when both cells
          and caption are evidence.
        """
        if element_id.startswith("table_caption_"):
            parts = element_id.split("_")
            return "table_" + parts[2] if len(parts) > 2 else "table_?"
        return "table_?"

    def _group_evidence_by_source(self, evidences: list[str]) -> dict[str, dict]:
        """Group evidence by wiki_doc and type.

        Format:
            {wiki_doc: {"sentences": [ev_id, ...], "cells": [(ev_id, table_id), ...],
             "items": [ev_id, ...], "captions": [(ev_id, table_id), ...]}}

        Rationale:
        - Group by page: load each page once and render all evidence from it.
        - Separate by type: sentences, tables (cells/captions), lists have different
          render logic and presentation order (passages first, then tables, then items).
        - Cells and captions store table_id for deduplication and table lookup.
        """
        by_source: dict[str, dict[str, list]] = defaultdict(
            lambda: {"sentences": [], "cells": [], "items": [], "captions": []}
        )
        for ev_id in evidences:
            parts = ev_id.split("_", 1)
            if len(parts) < 2:
                continue
            wiki_doc, element_id = parts[0], parts[1]
            if "sentence_" in element_id:
                by_source[wiki_doc]["sentences"].append(ev_id)
            elif "cell_" in element_id or "header_cell_" in element_id:
                table_id = self._parse_table_id_from_cell_id(element_id)
                by_source[wiki_doc]["cells"].append((ev_id, table_id))
            elif "table_caption_" in element_id:
                table_id = self._parse_table_id_from_caption_id(element_id)
                by_source[wiki_doc]["captions"].append((ev_id, table_id))
            elif "item_" in element_id:
                by_source[wiki_doc]["items"].append(ev_id)
        return dict(by_source)

    def _format_evidence_str(self, evidences: list[str]) -> str:
        """Build structured evidence string from grouped evidence.

        Format:
            [Source: <page title>]

            Passages:
            - <sentence 1>
            - <sentence 2>

            Table: <caption or empty>
            | col1 | col2 |
            | --- | --- |
            | ... |

            List items:
            - <item 1>
            - <item 2>

            (Sections separated by "\\n\\n". Multiple pages separated by "\\n\\n".)

        Rationale:
        - Source first: helps model attribute evidence to the right page.
        - Passages before tables before items: matches typical reading order.
        - One table per table_id: cells and captions from same table shown once.
        - Caption-only tables: when evidence is only table_caption, still show full
          table so the model has context.
        - Bold marking: evidence cells/sentences/items are **bold** so the model
          knows which parts support the claim (vs context).
        """
        grouped = self._group_evidence_by_source(evidences)
        sections = []
        for wiki_doc, ev_dict in grouped.items():
            page_json = self.wiki_db.get_doc_json(wiki_doc)
            wiki_page = WikiPage(wiki_doc, page_json)
            title = wiki_to_plain_text(wiki_page.get_title_content())
            parts = [f"[Source: {title}]", ""]

            if ev_dict["sentences"]:
                parts.append("Passages:")
                for ev_id in ev_dict["sentences"]:
                    element_id = ev_id.split("_", 1)[1]
                    elem = wiki_page.get_element_by_id(element_id)
                    if elem:
                        content = wiki_to_plain_text(str(elem))
                        parts.append(f"- **{content}**")
                parts.append("")

            table_evidence_cells: dict[str, set[str]] = defaultdict(set)
            for ev_id, table_id in ev_dict["cells"]:
                element_id = ev_id.split("_", 1)[1]
                table_evidence_cells[table_id].add(element_id)

            caption_evidence_tables = {t_id for _, t_id in ev_dict["captions"]}

            tables_done = set()
            if ev_dict["cells"]:
                for ev_id, table_id in ev_dict["cells"]:
                    if table_id in tables_done:
                        continue
                    tables_done.add(table_id)
                    element_id = ev_id.split("_", 1)[1]
                    table = wiki_page.get_table_from_cell_id(element_id)
                    if table:
                        caption = wiki_to_plain_text(table.get_table_caption())
                        if caption and table_id in caption_evidence_tables:
                            table_label = f"Table: **{caption}**"
                        else:
                            table_label = f"Table: {caption}" if caption else "Table"
                        parts.append(table_label)
                        evidence_cell_ids = table_evidence_cells[table_id]
                        parts.append(
                            self._format_table_as_markdown(
                                wiki_page, table, evidence_cell_ids
                            )
                        )
                        parts.append("")

            if ev_dict["captions"]:
                for ev_id, table_id in ev_dict["captions"]:
                    if table_id in tables_done:
                        continue
                    tables_done.add(table_id)
                    element_id = ev_id.split("_", 1)[1]
                    table = wiki_page.get_element_by_id(table_id)
                    if table:
                        caption = wiki_to_plain_text(
                            wiki_page.get_caption_content(element_id) or ""
                        )
                        table_label = (
                            f"Table: **{caption}**" if caption else "Table"
                        )
                        parts.append(table_label)
                        parts.append(
                            self._format_table_as_markdown(wiki_page, table)
                        )
                        parts.append("")

            if ev_dict["items"]:
                parts.append("List items:")
                for ev_id in ev_dict["items"]:
                    element_id = ev_id.split("_", 1)[1]
                    content = wiki_page.get_item_by_id(element_id)
                    if content is not None:
                        parts.append(
                            f"- **{wiki_to_plain_text(str(content))}**"
                        )
                parts.append("")

            sections.append("\n".join(parts).strip())
        return "\n\n".join(sections)

    def _format_context_str(self, context_dicts: dict) -> str:
        """Format context with plain text.

        Format:
            One line per context element, newline-separated.
            Content is normalized via wiki_to_plain_text (no markup).

        Supports: title, cell, header_cell, item, table_caption, sentence.
        Rationale:
        - Context provides supporting elements (e.g. section headers, surrounding
          cells); flattening to lines keeps it compact.
        - Same element-type dispatch as evidence: title/table_caption need
          specialized getters; sentence uses get_element_by_id.
        """
        parts = []
        for ev_key, contexts in context_dicts.items():
            for ctx_id in contexts:
                wiki_doc = ctx_id.split("_")[0]
                element_id = "_".join(ctx_id.split("_")[1:])
                page_json = self.wiki_db.get_doc_json(wiki_doc)
                wiki_page = WikiPage(wiki_doc, page_json)
                if "title" in element_id:
                    content = wiki_page.get_title_content()
                elif "cell" in element_id:
                    content = wiki_page.get_cell_content(element_id)
                elif "item" in element_id:
                    content = wiki_page.get_item_by_id(element_id)
                elif "table_caption_" in element_id:
                    content = wiki_page.get_caption_content(element_id)
                elif "sentence_" in element_id:
                    elem = wiki_page.get_element_by_id(element_id)
                    content = str(elem) if elem else ""
                else:
                    content = ""
                if content:
                    parts.append(wiki_to_plain_text(str(content)))
        return "\n".join(parts) if parts else ""

    def __iter__(self):
        for annotation in self.annotations:
            claim = annotation.get_claim()
            try:
                label = normalize_feverous_label(annotation.get_verdict())
                context_dicts = annotation.get_context(flat=True)
                evidences = annotation.get_evidence(flat=True)

                evidence = self._format_evidence_str(evidences)
                context = self._format_context_str(context_dicts)
            except Exception:
                context = None
                evidence = None
                label = None

            yield {
                "context": context,
                "claim": claim,
                "evidence": evidence,
                "label": label,
            }