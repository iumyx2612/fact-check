import multiprocessing as mp
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.indices import SummaryIndex
from llama_index.retrievers.bm25 import BM25Retriever

from src.modules.datasets.feverous.database.feverous_db import FeverousDB
from src.modules.datasets.feverous.utils.wiki_page import WikiPage


document_path = "datas/feverous/feverous_wikiv1.db"
persist_path = "output/graph_check"
workers = 4

db = FeverousDB(document_path)
doc_ids = db.get_doc_ids()


def task(doc_id):
    page_json = db.get_doc_json(doc_id)
    wiki_page = WikiPage(doc_id, page_json)
    document = Document(text=str(wiki_page))
    return document


if __name__ == '__main__':
    with mp.Pool(processes=workers) as pool:
        documents = list(
            tqdm(
                pool.imap_unordered(task, doc_ids, chunksize=100),
                total=len(doc_ids),
                desc="Indexing documents"
            )
        )

    index = SummaryIndex(nodes=documents)
    retriever = BM25Retriever.from_defaults(index, similarity_top_k=10)

    retriever.persist(persist_path)