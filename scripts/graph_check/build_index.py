import multiprocessing as mp
from pathlib import Path
import sys

from tqdm import tqdm
from pymilvus import MilvusException
from llama_index.core import Document
from llama_index.core.utils import iter_batch
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction


# Ensure project root is on sys.path so `src` imports work when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.datasets.feverous.database.feverous_db import FeverousDB
from src.modules.datasets.feverous.utils.feveous_utils import wiki_to_plain_text
from src.modules.datasets.feverous.utils.wiki_page import WikiPage, WikiSection, WikiTable
from src.utils.constants import MILVUS_URI, MILVUS_TOKEN, FEVEROUS_MILVUS_COLLECTION


DOCUMENT_PATH = "datas/feverous/feverous_wikiv1.db"
URI = MILVUS_URI
# DOC_IDS_BATCH_SIZE controls how many Feverous doc_ids are processed
# per multiprocessing batch.
DOC_IDS_BATCH_SIZE = 1000
# BATCH_SIZE controls how many Document objects are upserted to Milvus per call.
BATCH_SIZE = 100
WORKERS = 4
TOKEN = MILVUS_TOKEN

db = FeverousDB(DOCUMENT_PATH)
# Build Milvus vector store creating a new collection
vector_store = MilvusVectorStore(
    uri=URI,
    collection_name=FEVEROUS_MILVUS_COLLECTION,
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
    total_doc_ids = len(doc_ids)
    print(f"Found {total_doc_ids:,} Feverous documents to index.")

    processed_docs = 0
    indexed_docs = 0
    num_batches = (
        (total_doc_ids + DOC_IDS_BATCH_SIZE - 1) // DOC_IDS_BATCH_SIZE
        if total_doc_ids > 0
        else 0
    )

    # Use a single Pool for the whole run to avoid per-batch process spawn cost
    with mp.Pool(processes=WORKERS) as pool:
        for doc_ids_batch in tqdm(
            iter_batch(doc_ids, DOC_IDS_BATCH_SIZE),
            total=num_batches,
            desc="",
            bar_format="{l_bar}{bar}{r_bar}",
        ):
            documents = []

            # Choose a chunksize that balances scheduling overhead and load balancing
            chunksize = max(1, len(doc_ids_batch) // (WORKERS * 4))

            for result in tqdm(
                pool.imap_unordered(task, doc_ids_batch, chunksize=chunksize),
                total=len(doc_ids_batch),
                desc="",
                bar_format="{l_bar}{bar}{r_bar}",
            ):
                documents.extend(result)

            for batch_document in tqdm(
                iter_batch(documents, BATCH_SIZE),
                desc="",
                bar_format="{l_bar}{bar}{r_bar}",
            ):
                try:
                    vector_store.add(batch_document)
                except MilvusException:
                    vector_store.add(batch_document)
                indexed_docs += len(batch_document)

            processed_docs += len(doc_ids_batch)
            print(f"Processed {processed_docs:,} / {total_doc_ids:,} Feverous documents")

    print(
        f"Finished indexing Feverous: total unique pages processed = {processed_docs:,}"
    )
    print(f"Total Document objects upserted to Milvus = {indexed_docs:,}")