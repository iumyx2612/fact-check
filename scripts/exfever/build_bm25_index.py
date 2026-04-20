#!/usr/bin/env python3
"""
Build BM25 retriever for EX-FEVER dataset and persist to disk.
This separates index building from the benchmark flow to avoid OOM errors.
"""

import sys
from pathlib import Path

# Add the fact-check directory to the Python path so we can import src.modules
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
from src.modules.retrievers.exfever import build_exfever_retriever

# Configuration
# For testing, use only the small database first
EXFEVER_DB_PATHS = [
    "datas/ex-fever/wiki_db.db"
    # "datas/ex-fever/wiki_wo_links.db"  # Commented out for testing
]
PERSIST_DIR = "cache/exfever_bm25_retriever_small"
SIMILARITY_TOP_K = 10
NUM_WORKERS = 6
BATCH_SIZE = 10000
SKIP_STEMMING = True

def main():
    """Build and persist BM25 retriever for EX-FEVER dataset."""
    print("Building BM25 retriever for EX-FEVER dataset...")
    print(f"Database paths: {EXFEVER_DB_PATHS}")
    print(f"Persist directory: {PERSIST_DIR}")
    
    # Remove existing persist directory to build a fresh index
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
    
    # Do not create the persist directory here; let the build function create it
    # when persisting the retriever.
    
    # Build retriever (this will load all documents and build the index)
    retriever = build_exfever_retriever(
        db_path=EXFEVER_DB_PATHS,
        similarity_top_k=SIMILARITY_TOP_K,
        persist_dir=PERSIST_DIR,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        skip_stemming=SKIP_STEMMING
    )
    
    print(f"BM25 retriever built and persisted to {PERSIST_DIR}")
    print("You can now use this persisted retriever in benchmark_exfever_graphcheck.py")
    print("by setting --persist_dir=", PERSIST_DIR)

if __name__ == '__main__':
    main()