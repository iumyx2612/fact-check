from typing import Optional

from workflows import Workflow, step
from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices import SummaryIndex
from llama_index.retrievers.bm25 import BM25Retriever

from src.modules.datasets.feverous.database.feverous_db import FeverousDB
from src.modules.datasets.feverous.utils.wiki_page import WikiPage


def build_retriever(document_path: str) -> BM25Retriever:
    db = FeverousDB(document_path)
    doc_ids = db.get_doc_ids()

    documents = []
    for doc_id in doc_ids:
        page_json = db.get_doc_json(doc_id)
        wiki_page = WikiPage(doc_id, page_json)
        document = Document(text=str(wiki_page))
        documents.append(document)

    index = SummaryIndex(nodes=documents)
    retriever = BM25Retriever.from_defaults(index)

    return retriever


class GraphCheckWorkflow(Workflow):
    def __init__(
            self,
            llm: LLM,
            retriever: Optional[BaseRetriever] = None,
            document_path: str = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm
        if sum(bool(val) for val in [document_path, retriever]) != 1:
            raise ValueError("Please pass exactly one of document_path or retriever.")

        if document_path:
            retriever = build_retriever(document_path)

        self.retriever = retriever