import multiprocessing as mp
from tqdm import tqdm
from pymilvus import MilvusException
from llama_index.core import Document
from llama_index.core.utils import iter_batch
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

from src.modules.datasets.feverous.database.feverous_db import FeverousDB
from src.modules.datasets.feverous.utils.feveous_utils import wiki_to_plain_text
from src.modules.datasets.feverous.utils.wiki_page import WikiPage, WikiSection, WikiTable


DOCUMENT_PATH = "datas/feverous/feverous_wikiv1.db"
URI = "https://milvus.vm.trungtd.work:19530"
BATCH_SIZE = 100
WORKERS = 4
TOKEN = "root:Milvus"

db = FeverousDB(DOCUMENT_PATH)
# Build Milvus vector store creating a new collection
vector_store = MilvusVectorStore(
    uri=URI,
    collection_name="feverous_bm25",
    token=TOKEN,
    enable_dense=False,
    enable_sparse=True,  # Only enable sparse to demo full text search
    sparse_embedding_function=BM25BuiltInFunction(),
    overwrite=True,
    use_async_client=False,
)

def task(doc_id):
    page_json = db.get_doc_json(doc_id)
    wiki_page = WikiPage(doc_id, page_json)

    elements = wiki_page.get_page()
    title = wiki_to_plain_text(str(wiki_page.title))

    sections = []
    current_section = f"{title}\n"
    for element in elements:
        if isinstance(element, WikiTable): # skip table
            continue
        elif isinstance(element, WikiSection):
            sections.append(current_section)
            current_section = f"{title}\n"
            current_section += f"# {wiki_to_plain_text(str(element))}\n"
        else:
            current_section += f"{wiki_to_plain_text(str(element))} "
    sections.append(current_section) # add the last section

    documents = []
    for section in sections:
        documents.append(Document(text=section))
    return documents


if __name__ == '__main__':
    doc_ids = db.get_doc_ids()

    for doc_ids_batch in iter_batch(doc_ids, 1000):
        documents = []
        with mp.Pool(processes=WORKERS) as pool:
            for result in tqdm(
                    pool.imap_unordered(task, doc_ids_batch, chunksize=10),
                    total=len(doc_ids_batch),
                    desc="Indexing documents batch..."
            ):
                documents.extend(result)
        for batch_document in tqdm(iter_batch(documents, BATCH_SIZE), desc="Upsert batch..."):
            try:
                vector_store.add(documents)
            except MilvusException:
                vector_store.add(documents)