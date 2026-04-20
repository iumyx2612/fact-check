import logging
import multiprocessing as mp
from pathlib import Path
import sqlite3
import sys
import threading
import time
from queue import Queue
from typing import Iterable, List, Tuple, Union, Optional
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # Define psutil as None to avoid unbound variable
    logging.warning("psutil not available, memory monitoring disabled")

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
    "datas/ex-fever/wiki_db.db", # small database without links, use small for testing
    "datas/ex-fever/wiki_wo_links.db", # large database with links
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

# Memory management settings
MEMORY_THRESHOLD_MB = 8000  # 8GB threshold for reducing batch size
MEMORY_CHECK_INTERVAL = 50  # Check memory every 50 batches
MIN_BATCH_SIZE = 1000
MAX_BATCH_SIZE = 10000

# Checkpoint settings
def get_checkpoint_db_path() -> Path:
    return PROJECT_ROOT / "indexing_checkpoint.db"

CHECKPOINT_COMMIT_INTERVAL = 100  # Commit checkpoints every 100 batches


def init_checkpoint_db(overwrite: bool = False) -> Optional[sqlite3.Connection]:
    """Initialize checkpoint database for resume capability.
    
    Args:
        overwrite: If True, clear existing checkpoint data (fresh start)
        
    Returns:
        SQLite connection object or None if checkpointing disabled
    """
    checkpoint_path = get_checkpoint_db_path()
    if overwrite:
        # Remove existing checkpoint database for fresh start
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"Removed existing checkpoint database: {checkpoint_path}")
    
    conn = sqlite3.connect(checkpoint_path, timeout=30.0)
    conn.execute('''CREATE TABLE IF NOT EXISTS indexing_progress
                   (db_path TEXT, doc_id TEXT, processed_at TIMESTAMP,
                    PRIMARY KEY (db_path, doc_id))''')
    conn.commit()
    logger.info(f"Initialized checkpoint database: {checkpoint_path}")
    return conn


def get_processed_count(checkpoint_conn: Optional[sqlite3.Connection]) -> int:
    """Get count of already processed documents from checkpoint database."""
    if checkpoint_conn is None:
        return 0
    cursor = checkpoint_conn.execute("SELECT COUNT(*) FROM indexing_progress")
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def is_document_processed(checkpoint_conn: Optional[sqlite3.Connection], 
                          db_path: str, doc_id: str) -> bool:
    """Check if a document has already been processed."""
    if checkpoint_conn is None:
        return False
    cursor = checkpoint_conn.execute(
        "SELECT 1 FROM indexing_progress WHERE doc_id = ? LIMIT 1",
        (doc_id,)
    )
    result = cursor.fetchone() is not None
    cursor.close()
    return result


def update_checkpoint(checkpoint_conn: Optional[sqlite3.Connection], 
                     processed_pairs: List[Tuple[str, str]]) -> None:
    """Update checkpoint with newly processed document pairs."""
    if checkpoint_conn is None or not processed_pairs:
        return
        
    checkpoint_conn.executemany(
        "INSERT OR IGNORE INTO indexing_progress (db_path, doc_id, processed_at) VALUES (?, ?, datetime('now'))",
        processed_pairs
    )


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


def iter_unique_doc_ids(db_paths: Iterable[str], 
                       checkpoint_conn: Optional[sqlite3.Connection] = None) -> List[Tuple[str, str]]:
    """
    Collect unique (db_path, doc_id) pairs across all provided databases.
    
    Skips documents that have already been processed (if checkpointing enabled).

    This mirrors the behaviour of the previous in-memory BM25 builder,
    ensuring documents that appear in multiple DBs are only indexed once.
    """
    seen_doc_ids = set()
    all_pairs: List[Tuple[str, str]] = []

    for db_path in db_paths:
        doc_ids = get_doc_ids(db_path)
        for doc_id in doc_ids:
            # Skip if already processed (when checkpointing is enabled)
            if checkpoint_conn is not None and is_document_processed(checkpoint_conn, db_path, doc_id):
                continue
                
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                all_pairs.append((db_path, doc_id))

    return all_pairs


def get_current_memory_mb() -> float:
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE or psutil is None:
        return 0.0
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def adjust_batch_size(current_memory_mb: float, base_batch_size: int) -> int:
    """Adjust batch size based on current memory usage."""
    if not PSUTIL_AVAILABLE:
        return base_batch_size
        
    if current_memory_mb > MEMORY_THRESHOLD_MB:
        # Reduce batch size when memory usage is high
        adjusted = int(base_batch_size * 0.8)
        return max(MIN_BATCH_SIZE, adjusted)
    elif current_memory_mb < MEMORY_THRESHOLD_MB * 0.5:
        # Increase batch size when memory usage is low
        adjusted = int(base_batch_size * 1.2)
        return min(MAX_BATCH_SIZE, adjusted)
    else:
        return base_batch_size


def main(overwrite: bool = True, bm25_k1: float = 0.9, bm25_b: float = 0.4):
    """Build and persist BM25 retriever for EX-FEVER dataset with resume capability.

    Args:
        overwrite: Whether to overwrite existing collection
        bm25_k1: BM25 k1 parameter for the sparse index (term frequency saturation)
        bm25_b: BM25 b parameter for the sparse index (length normalization)
    """
    print(f"Starting EX-FEVER indexing (overwrite={overwrite})...")
    print(f"Using BM25 index parameters: k1={bm25_k1}, b={bm25_b}")

    # Initialize checkpoint database
    checkpoint_conn = init_checkpoint_db(overwrite=overwrite)

    # Resolve DB paths relative to project root so the script is cwd-independent.
    db_paths = [str(PROJECT_ROOT / p) for p in EXFEVER_DB_PATHS]

    # Collect all unique (db_path, doc_id) pairs, skipping already processed ones
    doc_pairs = iter_unique_doc_ids(db_paths, checkpoint_conn)
    total_doc_pairs = len(doc_pairs)
    already_processed = get_processed_count(checkpoint_conn)

    print(f"Found {total_doc_pairs:,} unique documents to index for ExFever.")
    if already_processed > 0:
        print(f"Skipping {already_processed:,} already processed documents (resume mode).")

    if total_doc_pairs == 0:
        print("All documents already processed. Nothing to do.")
        if checkpoint_conn:
            checkpoint_conn.close()
        return

    # Build Milvus vector store
    # When overwriting, we recreate the collection. When resuming, we reuse existing.
    vector_store = MilvusVectorStore(
        uri=URI,
        collection_name=COLLECTION_NAME,
        token=TOKEN,
        enable_dense=False,
        enable_sparse=True,  # Only sparse for BM25-style retrieval
        sparse_embedding_function=BM25BuiltInFunction(),  # type: ignore
        sparse_index_config={"bm25_k1": bm25_k1, "bm25_b": bm25_b},
        overwrite=overwrite,  # Crucial: only overwrite when starting fresh
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
                    store.add(batch_documents)  # type: ignore
                except MilvusException as exc:
                    logger.warning("Milvus insert failed (%s); retrying batch of %d.", exc, len(batch_documents))
                    store.add(batch_documents)  # type: ignore
                counter["count"] += len(batch_documents)
            except Exception as exc:
                logger.error("Upsert failed permanently for batch of %d: %s", len(batch_documents), exc)
                counter["errors"] = counter.get("errors", 0) + len(batch_documents)
            finally:
                q.task_done()

    upsert_thread = threading.Thread(
        target=upsert_worker,
        args=(upsert_queue, vector_store, indexed_docs_counter),
        daemon=True,
    )
    upsert_thread.start()

    # Index documents in batches using a long-lived multiprocessing pool.
    # Documents are streamed into the upsert queue as soon as they are ready,
    # so DB loading and Milvus upserts are overlapped.
    processed_pairs = 0
    batch_checkpoint_counter = 0
    doc_pairs_position = 0
    start_time = time.time()

    # Dynamic batch size tracking
    current_batch_size = BATCH_SIZE
    
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
            # Memory monitoring and dynamic batch adjustment
            if PSUTIL_AVAILABLE and processed_pairs % MEMORY_CHECK_INTERVAL == 0:
                current_memory_mb = get_current_memory_mb()
                logger.info(f"Memory usage: {current_memory_mb:.0f} MB")
                current_batch_size = adjust_batch_size(current_memory_mb, BATCH_SIZE)
                if current_batch_size != BATCH_SIZE:
                    logger.info(f"Adjusted batch size from {BATCH_SIZE} to {current_batch_size} based on memory usage")

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
                while len(current_batch) >= current_batch_size:
                    batch_documents = current_batch[:current_batch_size]
                    current_batch = current_batch[current_batch_size:]
                    upsert_queue.put(batch_documents)  # type: ignore

                    # Check memory after enqueuing to catch spikes during processing
                    if PSUTIL_AVAILABLE:
                        current_memory_mb = get_current_memory_mb()
                        if current_memory_mb > MEMORY_THRESHOLD_MB:
                            logger.warning(f"High memory usage detected after enqueue: {current_memory_mb:.0f} MB")
                            current_batch_size = adjust_batch_size(current_memory_mb, BATCH_SIZE)
                            if current_batch_size != BATCH_SIZE:
                                logger.info(f"Adjusted batch size from {BATCH_SIZE} to {current_batch_size} due to high memory")

            processed_pairs += len(doc_pairs_batch)
            batch_checkpoint_counter += len(doc_pairs_batch)
            doc_pairs_position += len(doc_pairs_batch)

            # Update checkpoint periodically
            if checkpoint_conn is not None and batch_checkpoint_counter >= CHECKPOINT_COMMIT_INTERVAL * DOC_PAIRS_BATCH_SIZE:
                # Get the recently processed pairs for checkpointing
                start_index = doc_pairs_position - batch_checkpoint_counter
                end_index = doc_pairs_position
                recently_processed = doc_pairs[start_index:end_index]
                update_checkpoint(checkpoint_conn, recently_processed)
                checkpoint_conn.commit()
                batch_checkpoint_counter = 0
                logger.info(f"Checkpoint updated. Processed {processed_pairs:,}/{total_doc_pairs:,} pairs")
            
            print(
                f"Processed {processed_pairs:,} / {total_doc_pairs:,} unique ExFever pages"
            )

        # Enqueue any remaining documents as a final partial batch.
        if current_batch:
            upsert_queue.put(current_batch)

    # Final checkpoint update
    if checkpoint_conn:
        # Get remaining uncheckpointed pairs
        start_index = max(0, doc_pairs_position - batch_checkpoint_counter)
        end_index = doc_pairs_position
        remaining_pairs = doc_pairs[start_index:end_index]
        if remaining_pairs:
            update_checkpoint(checkpoint_conn, remaining_pairs)
            checkpoint_conn.commit()
        checkpoint_conn.close()

    # Signal the upsert worker to finish and wait for all queued upserts.
    upsert_queue.put(None)
    upsert_queue.join()
    upsert_thread.join()

    elapsed_time = time.time() - start_time
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
    print(f"Total elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build EX-FEVER BM25 index with resume capability')
    parser.add_argument('--overwrite', action='store_true', default=True,
                        help='Overwrite existing index (default: True)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from previous checkpoint (overwrites --overwrite)')
    parser.add_argument('--bm25-k1', type=float, default=0.9,
                        help='BM25 k1 parameter for sparse index (default: 0.9)')
    parser.add_argument('--bm25-b', type=float, default=0.4,
                        help='BM25 b parameter for sparse index (default: 0.4)')
    args = parser.parse_args()

    # Handle resume when no checkpoint exists
    checkpoint_path = get_checkpoint_db_path()
    if args.resume:
        overwrite = not checkpoint_path.exists()
    else:
        overwrite = args.overwrite

    main(overwrite=overwrite, bm25_k1=args.bm25_k1, bm25_b=args.bm25_b)