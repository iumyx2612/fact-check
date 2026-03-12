"""
ExFever GraphCheck Benchmark

This script runs the full GraphCheck flow for ExFever dataset:
1. Graph Construction - decompose claim into latent entities and triples
2. Path Generation - generate valid orderings for entity resolution
3. Infilling - resolve latent entities using retrieval + LLM
4. Verification - verify each triple against evidence

Run with:
    cd /home/anhm/code/paper/src/fact-check && uv run python benchmark_exfever_graphcheck.py
"""
import os
import asyncio
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import logging
import time

from llama_index.llms.openai import OpenAI

from src.modules.datasets.exfever import ExFever
from src.modules.schema.graph_check.graph import Graph
from src.impls.workflows.graph_check.construct_graph import GraphConstructWorkflow
from src.impls.workflows.graph_check.infilling import InfillingWorkflow
from src.impls.workflows.graph_check.verification import VerificationWorkflow
from src.impls.events.graph_check.construct_graph import ConstructGraphStartEvent
from src.impls.events.graph_check.infilling import InfillingStartEvent, InfillingLoopInitialize, InfillingStopEvent
from src.impls.events.graph_check.verification import VerificationStartEvent
from src.modules.evaluator import evaluate_file

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set verbose loggers to DEBUG level
logging.getLogger("llama_index").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

load_dotenv()


EXFEVER_DATA_PATH = "datas/ex-fever/dev.csv"
EXFEVER_DB_PATHS = [
    "datas/ex-fever/wiki_db.db",
    "datas/ex-fever/wiki_wo_links.db"
]
OUTPUT_FILE = "result/exfever-graphcheck.csv"
PERSIST_DIR = "cache/bm25_retriever"
PATH_LIMIT = 5
SIMILARITY_TOP_K = 10

# Rate limiting: maximum concurrent API calls
MAX_CONCURRENT_REQUESTS = 10


def normalize_prediction(prediction: str) -> str:
    """Normalize prediction to match dataset label format."""
    prediction_mapping = {
        "SUPPORTED": "SUPPORT",
        "NOT_SUPPORTED": "REFUTE",
        "NOT_ENOUGH_INFORMATION": "NEI"
    }
    return prediction_mapping.get(prediction, prediction)


async def run_graphcheck_for_sample(
    llm: OpenAI,
    claim: str,
    retriever,
    path_limit: int = 5,
    semaphore: asyncio.Semaphore | None = None
) -> dict:
    """Run full GraphCheck flow for a single claim."""
    start_time = time.time()
    total_llm_calls = 0

    if semaphore:
        async with semaphore:
            return await _run_graphcheck_for_sample_impl(
                llm, claim, retriever, path_limit, start_time, total_llm_calls
            )
    else:
        return await _run_graphcheck_for_sample_impl(
            llm, claim, retriever, path_limit, start_time, total_llm_calls
        )


async def _run_graphcheck_for_sample_impl(
    llm: OpenAI,
    claim: str,
    retriever,
    path_limit: int,
    start_time: float,
    total_llm_calls: int
) -> dict:
    """Implementation of GraphCheck flow for a single claim."""
    try:
        logger.info(f"Processing claim: {claim[:80]}...")

        # Graph Construction
        construct_start = time.time()
        construct_wf = GraphConstructWorkflow(llm=llm)
        graph_result = await construct_wf.run(
            start_event=ConstructGraphStartEvent(claim=claim)
        )
        construct_time = time.time() - construct_start
        total_llm_calls += 1

        graph_obj = graph_result.graph if hasattr(graph_result, 'graph') else graph_result
        logger.debug(f"Graph constructed with {graph_obj.num_la_ent} latent entities, {len(graph_obj.total_triples)} triples")

        if graph_obj.num_la_ent == 0:
            logger.debug("No latent entities, skipping infilling")
            verification_wf = VerificationWorkflow(
                llm=llm,
                retriever=retriever,
                dataset_type="exfever"
            )
            result = await verification_wf.run(
                start_event=VerificationStartEvent(graph=graph_obj)
            )
            total_llm_calls += len(result.verification_results)
            normalized_pred = normalize_prediction(result.prediction)
            elapsed = time.time() - start_time
            logger.info(f"Result: {normalized_pred} | Triples: {len(graph_obj.total_triples)} | "
                       f"LLM calls: {total_llm_calls} | Time: {elapsed:.1f}s")
            return {
                "prediction": normalized_pred,
                "verification_results": result.verification_results,
                "graph": graph_obj
            }

        # Path Generation
        paths = graph_obj.get_valid_paths(path_limit)
        logger.debug(f"Generated {len(paths)} valid paths")

        if not paths:
            paths = [[]]

        # Try each path
        for path_idx, path in enumerate(paths):
            logger.debug(f"Trying path {path_idx + 1}/{len(paths)}: {path}")

            # Infilling
            infilling_start = time.time()
            infilling_wf = InfillingWorkflow(
                llm=llm,
                retriever=retriever,
                dataset_type="exfever"
            )

            infilling_result = await infilling_wf.run(
                start_event=InfillingStartEvent(claim=claim, path=path, graph=graph_obj)
            )
            infilling_time = time.time() - infilling_start
            total_llm_calls += len(path)

            infilled_graph = infilling_result.graph

            # Verification
            verification_start = time.time()
            verification_wf = VerificationWorkflow(
                llm=llm,
                retriever=retriever,
                dataset_type="exfever"
            )

            result = await verification_wf.run(
                start_event=VerificationStartEvent(graph=infilled_graph)
            )
            verification_time = time.time() - verification_start
            total_llm_calls += len(result.verification_results)

            logger.debug(f"Infilling: {infilling_time:.1f}s | Verification: {verification_time:.1f}s | "
                        f"Triples verified: {len(result.verification_results)}")

            if result.prediction == "SUPPORTED":
                normalized_pred = normalize_prediction(result.prediction)
                elapsed = time.time() - start_time
                logger.info(f"Result: {normalized_pred} | Paths tried: {path_idx + 1}/{len(paths)} | "
                           f"Triples: {len(infilled_graph.total_triples)} | LLM calls: {total_llm_calls} | "
                           f"Time: {elapsed:.1f}s")
                return {
                    "prediction": normalized_pred,
                    "verification_results": result.verification_results,
                    "path": path,
                    "graph": infilled_graph
                }

        # All paths failed
        normalized_pred = normalize_prediction("NOT_SUPPORTED")
        elapsed = time.time() - start_time
        logger.info(f"Result: {normalized_pred} | All {len(paths)} paths failed | "
                   f"Triples: {len(graph_obj.total_triples)} | LLM calls: {total_llm_calls} | "
                   f"Time: {elapsed:.1f}s")
        return {
            "prediction": normalized_pred,
            "verification_results": [],
            "paths_tried": paths,
            "graph": graph_obj
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing claim '{claim[:50]}...': {e} | Time: {elapsed:.1f}s", exc_info=True)
        return {
            "prediction": "REFUTE",
            "error": str(e)
        }


async def benchmark(
    data_path: str,
    db_path: str | List[str],
    output_file: str,
    model_name: str = "gpt-4.1-mini",
    path_limit: int = 5,
    similarity_top_k: int = 10,
    max_samples: int | None = None,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    persist_dir: str | None = None
):
    """
    Run GraphCheck benchmark on ExFever dataset with parallel processing.

    Args:
        data_path: Path to ExFever CSV file
        db_path: Path to ExFever wiki database, or list of paths to merge
        output_file: Output CSV file path
        model_name: OpenAI model to use
        path_limit: Maximum number of paths to try
        similarity_top_k: Number of documents to retrieve
        max_samples: Maximum number of samples to process (None for all)
        max_concurrent: Maximum number of concurrent API requests
        persist_dir: Directory to persist/load BM25 retriever (None to disable persistence)
    """

    logger.info(f"Loading dataset from {data_path}...")
    dataset = ExFever.from_csv(data_path)

    if max_samples:
        dataset_size = min(len(dataset), max_samples)
    else:
        dataset_size = len(dataset)

    logger.info(f"Loaded {len(dataset)} samples, processing {dataset_size} with max_concurrent={max_concurrent}...")

    # Configure LLM with optimized settings
    llm = OpenAI(
        model=model_name,
        max_retries=3,
        timeout=120.0,
        reuse_client=False  # Better for high-volume async calls
    )

    # Build retriever once and reuse across all samples
    logger.info("Building BM25 retriever (this may take a moment)...")
    from src.modules.retrievers.exfever import build_exfever_retriever
    retriever = build_exfever_retriever(
        db_path,
        similarity_top_k,
        persist_dir,
        num_workers=6,
        batch_size=10000,
        skip_stemming=True
    )
    logger.info("Retriever built successfully")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process samples in parallel
    async def process_sample(i: int) -> dict:
        sample = dataset[i]
        claim = sample["claim"]
        label = sample["label"]
        explanation = sample.get("explanation", "")

        result = await run_graphcheck_for_sample(
            llm=llm,
            claim=claim,
            retriever=retriever,
            path_limit=path_limit,
            semaphore=semaphore
        )

        prediction = result.get("prediction", "NOT_SUPPORTED")

        return {
            "claim": claim,
            "explanation": explanation,
            "label": label,
            "pred": prediction,
            "is_correct": prediction == label
        }

    # Run all samples in parallel with semaphore limiting concurrency
    start_time = time.time()
    results = await asyncio.gather(
        *[process_sample(i) for i in range(dataset_size)]
    )
    total_time = time.time() - start_time

    logger.info(f"Processed {dataset_size} samples in {total_time:.1f}s ({dataset_size/total_time:.2f} samples/s)")

    out_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run GraphCheck on ExFever dataset")
    parser.add_argument("--data_path", type=str, default=EXFEVER_DATA_PATH,
                        help="Path to ExFever CSV file")
    parser.add_argument("--db_path", type=str, nargs='+', default=EXFEVER_DB_PATHS,
                        help="Path(s) to ExFever wiki database(s). Can specify multiple to merge.")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output CSV file")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model to use")
    parser.add_argument("--path_limit", type=int, default=PATH_LIMIT,
                        help="Maximum number of paths to try")
    parser.add_argument("--top_k", type=int, default=SIMILARITY_TOP_K,
                        help="Number of documents to retrieve")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--max_concurrent", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help="Maximum number of concurrent API requests (rate limiting)")
    parser.add_argument("--persist_dir", type=str, default=PERSIST_DIR,
                        help="Directory to persist/load BM25 retriever (empty string to disable)")

    args = parser.parse_args()

    persist_dir = args.persist_dir if args.persist_dir else None

    output_file = asyncio.run(benchmark(
        data_path=args.data_path,
        db_path=args.db_path,
        output_file=args.output,
        model_name=args.model,
        path_limit=args.path_limit,
        similarity_top_k=args.top_k,
        max_samples=args.max_samples,
        max_concurrent=args.max_concurrent,
        persist_dir=persist_dir
    ))

    logger.info("Running evaluation...")
    evaluate_file(output_file)
