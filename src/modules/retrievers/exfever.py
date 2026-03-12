import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from llama_index.core import Document
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever


def normalize(text: str) -> str:
    """Resolve different type of unicode encodings."""
    import unicodedata
    return unicodedata.normalize('NFD', text)


def _load_document_batch(args: tuple) -> List[Document]:
    """
    Load a batch of documents from database (worker function for multiprocessing).

    Args:
        args: Tuple of (db_path, doc_ids)

    Returns:
        List of Document objects
    """
    import sqlite3
    from llama_index.core import Document

    db_path, doc_ids = args

    connection = sqlite3.connect(db_path, check_same_thread=False)
    cursor = connection.cursor()

    documents = []
    for doc_id in doc_ids:
        cursor.execute("SELECT text FROM documents WHERE id = ?", (normalize(doc_id),))
        result = cursor.fetchone()
        if result and result[0]:
            documents.append(Document(text=result[0], metadata={"doc_id": doc_id}))

    cursor.close()
    connection.close()
    return documents


def build_exfever_retriever(
    db_path: str | List[str],
    similarity_top_k: int = 10,
    persist_dir: Optional[str] = None,
    num_workers: int = 4,
    batch_size: int = 10000,
    skip_stemming: bool = True
) -> BM25Retriever:
    """
    Build a BM25 retriever for EX-FEVER dataset.

    Args:
        db_path: Path to EX-FEVER wiki_db.db SQLite database, or list of database paths
                 to merge (e.g., [wiki_db.db, wiki_wo_links.db])
        similarity_top_k: Number of top results to return
        persist_dir: Optional directory path to persist/load retriever. If provided and
                     directory exists, loads from disk instead of building. Otherwise,
                     builds and persists to disk.
        num_workers: Number of parallel workers for document loading (default: 4)
        batch_size: Number of documents to load per batch (default: 10000)
        skip_stemming: Skip stemming for faster tokenization (default: True, slight quality trade-off)

    Returns:
        BM25Retriever instance for document retrieval
    """
    import sqlite3

    if persist_dir is not None:
        persist_path = Path(persist_dir)
        if persist_path.exists():
            print(f"Loading BM25 retriever from {persist_dir}...")
            return BM25Retriever.from_persist_dir(str(persist_path), mmap=True)

    if isinstance(db_path, str):
        db_paths = [db_path]
    else:
        db_paths = db_path

    def get_doc_ids(db_path: str) -> List[str]:
        """Fetch all ids of docs stored in the db."""
        connection = sqlite3.connect(db_path, check_same_thread=False)
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        connection.close()
        return results

    # Collect all doc_ids first (fast operation)
    print("Collecting document IDs...")
    seen_doc_ids = set()
    all_doc_ids = []

    for db_path in db_paths:
        doc_ids = get_doc_ids(db_path)
        for doc_id in doc_ids:
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                all_doc_ids.append((db_path, doc_id))

    print(f"Found {len(all_doc_ids):,} unique documents to load")

    # Group doc_ids by database for efficient batch processing
    db_batches = {}
    for db_path, doc_id in all_doc_ids:
        if db_path not in db_batches:
            db_batches[db_path] = []
        db_batches[db_path].append(doc_id)

    # Create batches for parallel processing
    batches = []
    for db_path, doc_ids in db_batches.items():
        for i in range(0, len(doc_ids), batch_size):
            batch = doc_ids[i:i+batch_size]
            batches.append((db_path, batch))

    print(f"Loading {len(batches)} batches with {num_workers} workers...")

    # Load documents in parallel using ThreadPoolExecutor (compatible with asyncio)
    documents = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(batches), desc="Loading documents") as pbar:
            futures = {executor.submit(_load_document_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch_docs = future.result()
                documents.extend(batch_docs)
                pbar.update(1)

    print(f"Loaded {len(documents):,} documents")

    # Build BM25 retriever with optimized settings
    print("Building BM25 index...")
    retriever = BM25Retriever.from_defaults(
        nodes=documents,
        similarity_top_k=similarity_top_k,
        verbose=False,
        skip_stemming=skip_stemming
    )
    print("BM25 index built successfully")

    if persist_dir is not None:
        persist_path = Path(persist_dir)
        print(f"Persisting BM25 retriever to {persist_dir}...")
        retriever.persist(str(persist_path))
        print("Retriever persisted successfully")

    return retriever


class ExFeverRetriever:
    """
    EX-FEVER document retriever using BM25.
    
    Provides retrieval from EX-FEVER Wikipedia corpus.
    """
    
    def __init__(
        self,
        db_path: Optional[str | List[str]] = None,
        retriever: Optional[BM25Retriever] = None,
        similarity_top_k: int = 10,
        persist_dir: Optional[str] = None,
        num_workers: int = 4,
        batch_size: int = 10000,
        skip_stemming: bool = True
    ):
        """
        Initialize EX-FEVER retriever.

        Args:
            db_path: Path to wiki_db.db or list of database paths to merge
                     (used to build retriever if not provided)
            retriever: Existing BM25Retriever instance (optional)
            similarity_top_k: Number of top results to return
            persist_dir: Optional directory path to persist/load retriever
            num_workers: Number of parallel workers for document loading (default: 4)
            batch_size: Number of documents to load per batch (default: 10000)
            skip_stemming: Skip stemming for faster tokenization (default: True)
        """
        if retriever is None and db_path is None:
            raise ValueError("Either db_path or retriever must be provided")

        if retriever is None:
            assert db_path is not None  # Type assertion after check
            self.retriever = build_exfever_retriever(
                db_path, similarity_top_k, persist_dir, num_workers, batch_size, skip_stemming
            )
        else:
            self.retriever = retriever

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve documents for a given query.

        Args:
            query: Search query string
            top_k: Number of results to return (overrides default)

        Returns:
            List of NodeWithScore objects with text and metadata
        """
        nodes = self.retriever.retrieve(query)

        if top_k is not None:
            return nodes[:top_k]
        return nodes

    def retrieve_texts(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve document texts for a given query.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of document text strings
        """
        nodes = self.retrieve(query, top_k)
        return [node.text for node in nodes]
