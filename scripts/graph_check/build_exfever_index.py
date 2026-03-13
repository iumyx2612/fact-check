import logging
import multiprocessing as mp
from pathlib import Path
import sqlite3
import sys
from threading import Thread
from queue import Queue
from typing import Iterable, List, Tuple, Union

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

from src.utils.constants import MILVUS_URI, MILVUS_TOKEN, EXFEVER_MILVUS_COLLECTION

logger = logging.getLogger(__name__)


# Paths to the ExFever Wikipedia SQLite databases
EXFEVER_DB_PATHS: List[str] = [
    "datas/ex-fever/wiki_db.db",
    "datas/ex-fever/wiki_wo_links.db",
]

# Milvus connection parameters (keep consistent with Feverous index)
URI = MILVUS_URI
TOKEN = MILVUS_TOKEN
COLLECTION_NAME = EXFEVER_MILVUS_COLLECTION

# Indexing configuration
# DOC_PAIRS_BATCH_SIZE controls how many (db_path, doc_id) pairs are loaded
# and processed per multiprocessing batch.
DOC_PAIRS_BATCH_SIZE = 5000
# BATCH_SIZE controls how many Document objects are upserted to Milvus per Milvus call.
# Combined with UPSERT_QUEUE_MAXSIZE this bounds the number of Documents that can be
# buffered in memory while waiting to be upserted.
BATCH_SIZE = 4000
# Maximum number of Milvus batches waiting in memory.
UPSERT_QUEUE_MAXSIZE = 8
WORKERS = 32


def get_doc_ids(db_path: str) -> List[str]:
    """Fetch all document IDs from a given ExFever SQLite database."""
    connection = sqlite3.connect(db_path, check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM documents")
    results = [row[0] for row in cursor.fetchall()]
    cursor.close()
    connection.close()
    return results


def task(args: Tuple[str, str]) -> List[Document]:
    """
    Worker task to load a single document from the SQLite database
    and convert it into one or more `Document` objects.

    Args:
        args: Tuple of (db_path, doc_id)

    Returns:
        List of Document objects for this doc_id (usually length 1)
    """
    db_path, doc_id = args

    connection = sqlite3.connect(db_path, check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    cursor.close()
    connection.close()

    if not row or not row[0]:
        return []

    text = row[0]
    # For ExFever we index the full page text as a single document.
    return [Document(text=text, metadata={"doc_id": doc_id})]


def iter_unique_doc_ids(db_paths: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Collect unique (db_path, doc_id) pairs across all provided databases.

    This mirrors the behaviour of the previous in-memory BM25 builder,
    ensuring documents that appear in multiple DBs are only indexed once.
    """
    seen_doc_ids = set()
    all_pairs: List[Tuple[str, str]] = []

    for db_path in db_paths:
        doc_ids = get_doc_ids(db_path)
        for doc_id in doc_ids:
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                all_pairs.append((db_path, doc_id))

    return all_pairs


if __name__ == "__main__":
    # Resolve DB paths relative to project root so the script is cwd-independent.
    db_paths = [str(PROJECT_ROOT / p) for p in EXFEVER_DB_PATHS]

    # Collect all unique (db_path, doc_id) pairs
    doc_pairs = iter_unique_doc_ids(db_paths)
    total_doc_pairs = len(doc_pairs)
    print(f"Found {total_doc_pairs:,} unique documents to index for ExFever.")

    # Build Milvus vector store backed by BM25 sparse function.
    # overwrite=True ensures a clean collection rebuild.
    vector_store = MilvusVectorStore(
        uri=URI,
        collection_name=COLLECTION_NAME,
        token=TOKEN,
        enable_dense=False,
        enable_sparse=True,  # Only sparse for BM25-style retrieval
        sparse_embedding_function=BM25BuiltInFunction(),
        overwrite=True,
        use_async_client=False,
    )

    # Queue and background worker for Milvus upserts so that SQLite loading
    # and Milvus I/O can run in parallel.
    upsert_queue: "Queue[List[Document] | None]" = Queue(
        maxsize=UPSERT_QUEUE_MAXSIZE
    )
    indexed_docs_counter: dict[str, int] = {"count": 0, "skipped": 0, "errors": 0}

    # Pass queue/store/counter explicitly so the thread target does not rely on
    # closure lookup (avoids NameError if the nested function's scope is not
    # wired as expected when the worker runs).
    def upsert_worker(
        q: "Queue[Union[List[Document], None]]",
        store: MilvusVectorStore,
        counter: dict[str, int],
    ) -> None:
        while True:
            batch_documents = q.get()
            if batch_documents is None:
                q.task_done()
                break
            try:
                try:
                    store.add(batch_documents)
                except MilvusException as exc:
                    logger.warning("Milvus insert failed (%s); retrying batch of %d.", exc, len(batch_documents))
                    store.add(batch_documents)
                counter["count"] += len(batch_documents)
            except Exception as exc:
                logger.error("Upsert failed permanently for batch of %d: %s", len(batch_documents), exc)
                counter["errors"] = counter.get("errors", 0) + len(batch_documents)
            finally:
                q.task_done()

    upsert_thread = Thread(
        target=upsert_worker,
        args=(upsert_queue, vector_store, indexed_docs_counter),
        daemon=True,
    )
    upsert_thread.start()

    # Index documents in batches using a long-lived multiprocessing pool.
    # Documents are streamed into the upsert queue as soon as they are ready,
    # so DB loading and Milvus upserts are overlapped.
    processed_pairs = 0
    num_batches = (
        (total_doc_pairs + DOC_PAIRS_BATCH_SIZE - 1) // DOC_PAIRS_BATCH_SIZE
        if total_doc_pairs > 0
        else 0
    )
    print(f"Number of batches: {num_batches}")

    with mp.Pool(processes=WORKERS) as pool:
        current_batch: List[Document] = []

        for doc_pairs_batch in tqdm(
            iter_batch(doc_pairs, DOC_PAIRS_BATCH_SIZE),
            total=num_batches,
            desc="",
            bar_format="{l_bar}{bar}{r_bar}",
        ):
            # Choose a chunksize that balances scheduling overhead and load balancing
            chunksize = max(1, len(doc_pairs_batch) // (WORKERS * 4))

            for result in tqdm(
                pool.imap_unordered(task, doc_pairs_batch, chunksize=chunksize),
                total=len(doc_pairs_batch),
                desc="",
                bar_format="{l_bar}{bar}{r_bar}",
            ):
                if not result:
                    indexed_docs_counter["skipped"] += 1
                    continue
                current_batch.extend(result)

                # When we have enough documents for a Milvus batch, enqueue them.
                while len(current_batch) >= BATCH_SIZE:
                    batch_documents = current_batch[:BATCH_SIZE]
                    current_batch = current_batch[BATCH_SIZE:]
                    upsert_queue.put(batch_documents)

            processed_pairs += len(doc_pairs_batch)
            print(
                f"Processed {processed_pairs:,} / {total_doc_pairs:,} unique ExFever pages"
            )

        # Enqueue any remaining documents as a final partial batch.
        if current_batch:
            upsert_queue.put(current_batch)

    # Signal the upsert worker to finish and wait for all queued upserts.
    upsert_queue.put(None)
    upsert_queue.join()
    upsert_thread.join()

    indexed_docs = indexed_docs_counter["count"]
    skipped_docs = indexed_docs_counter["skipped"]
    failed_docs = indexed_docs_counter.get("errors", 0)
    print(
        f"Finished indexing ExFever: total unique pages processed = {processed_pairs:,}"
    )
    print(f"Total Document objects upserted to Milvus = {indexed_docs:,}")
    if skipped_docs:
        print(f"Skipped (empty text) = {skipped_docs:,}")
    if failed_docs:
        print(f"Failed (permanent Milvus error) = {failed_docs:,}")

