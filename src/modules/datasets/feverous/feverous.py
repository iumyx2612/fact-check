from collections import defaultdict
from typing import Optional

from .utils.annotation_processor import AnnotationProcessor
from .utils.wiki_page import WikiPage
from .database.feverous_db import FeverousDB
from ..base import Dataset
from .utils import normalize_feverous_label

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

        wiki_db = None
        if db_path:
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
                label = None
                context = None
                evidence = None

            yield {
                "context": context,
                "claim": claim,
                "evidence": evidence,
                "label": label
            }
