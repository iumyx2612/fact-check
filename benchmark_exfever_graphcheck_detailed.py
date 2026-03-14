"""
ExFever GraphCheck Benchmark with Detailed Flow Tracking

This script runs the full GraphCheck flow for ExFever dataset with detailed tracking
of each step for debugging purposes:
1. Graph Construction - decompose claim into latent entities and triples
2. Path Generation - generate valid orderings for entity resolution
3. Infilling - resolve latent entities using retrieval + LLM
4. Verification - verify each triple against evidence

Run with:
    cd /home/anhm/code/paper/src/fact-check && uv run python benchmark_exfever_graphcheck_detailed.py
"""
import os
import asyncio
import json
import random
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import logging
import time

from openai import RateLimitError

from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatMessage

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

load_dotenv('.env')


EXFEVER_DATA_PATH = "datas/ex-fever/dev.csv"
EXFEVER_DB_PATHS = [
    "datas/ex-fever/wiki_db.db",
    "datas/ex-fever/wiki_wo_links.db"
]
OUTPUT_FILE = "result/exfever-graphcheck-detailed.csv"
PATH_LIMIT = 5
SIMILARITY_TOP_K = 10

# Rate limiting: maximum concurrent API calls
MAX_CONCURRENT_REQUESTS = 1


async def retry_on_rate_limit(coro_fn, *args, max_retries: int = 8, base_sleep: float = 1.5, **kwargs):
    """Retry an async coroutine function on RateLimitError with exponential backoff.

    Args:
        coro_fn: Async callable to invoke.
        *args: Positional arguments forwarded to coro_fn.
        max_retries: Maximum number of retry attempts before re-raising.
        base_sleep: Base sleep duration in seconds; doubles with each attempt plus jitter.
        **kwargs: Keyword arguments forwarded to coro_fn.

    Returns:
        Result of coro_fn on success.

    Raises:
        RateLimitError: If all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            sleep_duration = base_sleep * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_retries}), "
                f"sleeping {sleep_duration:.1f}s before retry..."
            )
            await asyncio.sleep(sleep_duration)


def normalize_prediction(prediction: str) -> str:
    """Normalize prediction to match dataset label format."""
    prediction_mapping = {
        "SUPPORTED": "SUPPORT",
        "NOT_SUPPORTED": "REFUTE",
        "NOT_ENOUGH_INFORMATION": "NEI"
    }
    return prediction_mapping.get(prediction, prediction)


def serialize_verification_results(verification_results):
    """Serialize verification results for CSV storage with more details."""
    if not verification_results:
        return "[]"
    
    # Create a detailed version for CSV
    detailed_results = []
    for result in verification_results:
        detailed_result = {
            "triple": result.get("triple", ""),
            "prediction": result.get("prediction", ""),
            "evidence": result.get("retrieved_evidence", "")[:200],  # Truncate for CSV
            "evidence_length": len(result.get("retrieved_evidence", ""))
        }
        detailed_results.append(detailed_result)
    
    return json.dumps(detailed_results, ensure_ascii=False)


def extract_graph_details(graph_obj):
    """Extract detailed information from graph object for CSV storage."""
    if not graph_obj:
        return {}
    
    details = {
        "num_latent_entities": getattr(graph_obj, 'num_la_ent', 0),
        "num_triples": len(getattr(graph_obj, 'total_triples', [])),
        "has_latent_entities": getattr(graph_obj, 'num_la_ent', 0) > 0,
        "latent_entities": [],
        "triples": []
    }
    
    # Extract latent entities if available
    if hasattr(graph_obj, 'la_ent_2_def'):
        details["latent_entities"] = [
            {"entity": entity, "definition": definition} 
            for entity, definition in graph_obj.la_ent_2_def.items()
        ]
    
    # Extract triples if available
    if hasattr(graph_obj, 'total_triples'):
        details["triples"] = [
            {
                "sentence": getattr(triple, 'sentence', ''),
                "triplet_text": getattr(triple, 'triplet_text', ''),
                "subject": getattr(triple, 'subject', ''),
                "relation": getattr(triple, 'relation', ''),
                "object": getattr(triple, 'object', '')
            }
            for triple in graph_obj.total_triples
        ]
    
    return details


def extract_infilling_details(infilling_result):
    """Extract detailed information from infilling result."""
    if not infilling_result:
        return {}
    
    details = {
        "success": False,
        "entities_filled": [],
        "queries": [],
        "answers": [],
        "evidence_retrieved": []
    }
    
    if hasattr(infilling_result, 'graph'):
        details["success"] = True
        # Try to get infilling log from context if available
        # This would need to be passed through the result object
    
    return details


async def run_graphcheck_for_sample_detailed(
    llm: OpenAI,
    claim: str,
    label: str,
    retriever,
    path_limit: int = 5,
    semaphore: asyncio.Semaphore | None = None
) -> dict:
    """Run full GraphCheck flow for a single claim with detailed tracking."""
    start_time = time.time()
    
    # Initialize tracking dictionaries
    tracking = {
        "claim": claim,
        "graph_construction": {},
        "path_generation": {},
        "infilling": {},
        "verification": {},
        "final_prediction": "NOT_SUPPORTED",
        "is_correct": False,
        "error": None,
        "total_llm_calls": 0,
        "elapsed_time": 0,
        "path_results": []  # Store results for each path attempted
    }

    async def _run_single_sample() -> dict:
        nonlocal start_time, tracking
        try:
            logger.info(f"Processing claim: {claim[:80]}...")

            # Graph Construction
            construct_start = time.time()
            construct_wf = GraphConstructWorkflow(llm=llm)
            graph_result = await retry_on_rate_limit(
                construct_wf.run,
                start_event=ConstructGraphStartEvent(claim=claim)
            )
            construct_time = time.time() - construct_start
            tracking["total_llm_calls"] += 1

            graph_obj = graph_result.graph if hasattr(graph_result, 'graph') else graph_result
            tracking["graph_construction"] = {
                "time": construct_time,
                "details": extract_graph_details(graph_obj),
                "raw_result": str(graph_result)[:200]  # Truncate for storage
            }
            
            logger.debug(f"Graph constructed with {graph_obj.num_la_ent} latent entities, {len(graph_obj.total_triples)} triples")

            if graph_obj.num_la_ent == 0:
                logger.debug("No latent entities, skipping infilling")
                tracking["path_generation"] = {"paths": [], "count": 0}
                tracking["infilling"] = {"skipped": True, "reason": "no_latent_entities"}
                
                verification_wf = VerificationWorkflow(
                    llm=llm,
                    retriever=retriever,
                    dataset_type="exfever"
                )
                result = await retry_on_rate_limit(
                    verification_wf.run,
                    start_event=VerificationStartEvent(graph=graph_obj)
                )
                tracking["total_llm_calls"] += len(result.verification_results)
                normalized_pred = normalize_prediction(result.prediction)
                tracking["final_prediction"] = normalized_pred
                tracking["verification"] = {
                    "time": 0,  # Will be calculated below
                    "results": result.verification_results,
                    "details": extract_verification_details(result)
                }
                
                tracking["elapsed_time"] = time.time() - start_time
                logger.info(f"Result: {normalized_pred} | Triples: {len(graph_obj.total_triples)} | "
                           f"LLM calls: {tracking['total_llm_calls']} | Time: {tracking['elapsed_time']:.1f}s")
                return tracking

            # Path Generation
            paths = graph_obj.get_valid_paths(path_limit)
            tracking["path_generation"] = {
                "paths": paths,
                "count": len(paths)
            }
            logger.debug(f"Generated {len(paths)} valid paths")

            if not paths:
                paths = [[]]

            # Try each path
            path_results = []
            logger.debug(f"Starting path loop with {len(paths)} paths")
            for path_idx, path in enumerate(paths):
                logger.debug(f"Trying path {path_idx + 1}/{len(paths)}: {path}")
                logger.debug(f"Current path_results length before processing: {len(path_results)}")
                
                path_start = time.time()
                
                # Infilling
                infilling_start = time.time()
                infilling_wf = InfillingWorkflow(
                    llm=llm,
                    retriever=retriever,
                    dataset_type="exfever"
                )

                infilling_result = await retry_on_rate_limit(
                    infilling_wf.run,
                    start_event=InfillingStartEvent(claim=claim, path=path, graph=graph_obj)
                )
                infilling_time = time.time() - infilling_start
                tracking["total_llm_calls"] += len(path)  # Approximate LLM calls for infilling

                infilled_graph = infilling_result.graph
                
                path_infilling_details = {
                    "path": path,
                    "time": infilling_time,
                    "details": extract_infilling_details(infilling_result),
                    "success": infilling_result is not None and hasattr(infilling_result, 'graph')
                }

                # Verification
                verification_start = time.time()
                verification_wf = VerificationWorkflow(
                    llm=llm,
                    retriever=retriever,
                    dataset_type="exfever"
                )

                result = await retry_on_rate_limit(
                    verification_wf.run,
                    start_event=VerificationStartEvent(graph=infilled_graph)
                )
                verification_time = time.time() - verification_start
                tracking["total_llm_calls"] += len(result.verification_results)
                
                path_verification_details = {
                    "time": verification_time,
                    "results": result.verification_results,
                    "details": extract_verification_details(result),
                    "prediction": result.prediction
                }

                path_total_time = time.time() - path_start
                
                path_result = {
                    "path_index": path_idx,
                    "path": path,
                    "infilling": path_infilling_details,
                    "verification": path_verification_details,
                    "total_time": path_total_time,
                    "graph_details": extract_graph_details(infilled_graph),
                    "prediction": result.prediction,
                    "normalized_prediction": normalize_prediction(result.prediction)
                }
                
                path_results.append(path_result)
                tracking["path_results"] = path_results
                logger.debug(f"Path result appended. Total path_results length: {len(path_results)}")

                logger.debug(f"Infilling: {infilling_time:.1f}s | Verification: {verification_time:.1f}s | "
                            f"Triples verified: {len(result.verification_results)}")

                if result.prediction == "SUPPORTED":
                    normalized_pred = normalize_prediction(result.prediction)
                    elapsed = time.time() - start_time
                    tracking["final_prediction"] = normalized_pred
                    tracking["is_correct"] = normalized_pred == label
                    tracking["elapsed_time"] = elapsed
                    tracking["graph_construction"]["time"] = construct_time
                    tracking["path_generation"]["selected_path_index"] = path_idx
                    tracking["infilling"] = path_infilling_details
                    tracking["verification"] = path_verification_details
                    
                    logger.info(f"Result: {normalized_pred} | Paths tried: {path_idx + 1}/{len(paths)} | "
                               f"Triples: {len(infilled_graph.total_triples)} | LLM calls: {tracking['total_llm_calls']} | "
                               f"Time: {elapsed:.1f}s")
                    return tracking

            # All paths failed
            logger.debug(f"All paths failed. Final path_results length: {len(path_results)}")
            normalized_pred = normalize_prediction("NOT_SUPPORTED")
            elapsed = time.time() - start_time
            tracking["final_prediction"] = normalized_pred
            tracking["is_correct"] = normalized_pred == label
            tracking["elapsed_time"] = elapsed
            tracking["graph_construction"]["time"] = construct_time
            tracking["path_generation"]["selected_path_index"] = -1  # No successful path
            tracking["infilling"] = {"attempted_paths": len(paths), "all_failed": True}
            if tracking["path_results"]:
                tracking["verification"] = tracking["path_results"][-1]["verification"]  # Last attempt
            else:
                logger.debug(f"tracking['path_results'] is empty: {tracking.get('path_results', [])}")
                # Create empty verification structure when no paths were attempted
                tracking["verification"] = {
                    "time": 0,
                    "results": [],
                    "details": extract_verification_details(None)
                }
            
            logger.info(f"Result: {normalized_pred} | All {len(paths)} paths failed | "
                       f"Triples: {len(graph_obj.total_triples)} | LLM calls: {tracking['total_llm_calls']} | "
                       f"Time: {elapsed:.1f}s")
            return tracking

        except Exception as e:
            elapsed = time.time() - start_time
            tracking["error"] = str(e)
            tracking["elapsed_time"] = elapsed
            tracking["final_prediction"] = "REFUTE"  # Default on error
            logger.error(f"Error processing claim '{claim[:50]}...': {e} | Time: {elapsed:.1f}s", exc_info=True)
            return tracking

    # Enforce global concurrency limit if a semaphore is provided
    if semaphore is not None:
        async with semaphore:
            return await _run_single_sample()

    return await _run_single_sample()


def extract_verification_details(verification_result):
    """Extract detailed information from verification result."""
    details = {
        "triples_processed": 0,
        "support_count": 0,
        "refute_count": 0,
        "nei_count": 0,
        "final_prediction": "",
        "triple_details": []
    }
    
    if not verification_result:
        return details
    
    details["final_prediction"] = getattr(verification_result, 'prediction', '')
    
    if hasattr(verification_result, 'verification_results'):
        details["triples_processed"] = len(verification_result.verification_results)
        
        for triple_result in verification_result.verification_results:
            triple_detail = {
                "triple": triple_result.get("triple", ""),
                "prediction": triple_result.get("prediction", ""),
                "evidence_preview": triple_result.get("retrieved_evidence", "")[:100] if triple_result.get("retrieved_evidence") else ""
            }
            
            pred = triple_result.get("prediction", "")
            if pred == "SUPPORTED":
                details["support_count"] += 1
            elif pred == "NOT_SUPPORTED":
                details["refute_count"] += 1
            else:
                details["nei_count"] += 1
                
            details["triple_details"].append(triple_detail)
    
    return details


async def benchmark(
    data_path: str,
    db_path: str | List[str],
    output_file: str,
    model_name: str = "gpt-4.1-mini",
    path_limit: int = 5,
    similarity_top_k: int = 10,
    max_samples: int | None = None,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
):
    """
    Run GraphCheck benchmark on ExFever dataset with parallel processing and detailed tracking.

    Args:
        data_path: Path to ExFever CSV file
        db_path: Path to ExFever wiki database, or list of paths to merge
                 (kept for compatibility; not used with Milvus-based retrieval)
        output_file: Output CSV file path
        model_name: OpenAI model to use
        path_limit: Maximum number of paths to try
        similarity_top_k: Number of documents to retrieve
        max_samples: Maximum number of samples to process (None for all)
        max_concurrent: Maximum number of concurrent API requests
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

    # Build Milvus-backed retriever once and reuse across all samples
    logger.info("Connecting to Milvus-backed ExFever retriever...")
    from src.modules.retrievers.exfever import build_exfever_milvus_retriever

    retriever = build_exfever_milvus_retriever(
        similarity_top_k=similarity_top_k,
    )
    logger.info("Retriever ready")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process samples in parallel
    async def process_sample(i: int) -> dict:
        sample = dataset[i]
        claim = sample["claim"]
        label = sample["label"]
        explanation = sample.get("explanation", "")

        result = await run_graphcheck_for_sample_detailed(
            llm=llm,
            claim=claim,
            label=label,
            retriever=retriever,
            path_limit=path_limit,
            semaphore=semaphore
        )
        
        # Add label and correctness info
        result["label"] = label
        result["is_correct"] = result["final_prediction"] == label

        # Create flat dictionary for CSV output
        flat_result = {
            "claim": claim,
            "explanation": explanation,
            "label": label,
            "pred": result["final_prediction"],
            "is_correct": result["is_correct"],
            
            # Graph Construction Details
            "graph_num_latent_entities": result["graph_construction"]["details"]["num_latent_entities"],
            "graph_num_triples": result["graph_construction"]["details"]["num_triples"],
            "graph_has_latent_entities": result["graph_construction"]["details"]["has_latent_entities"],
            "construction_time": result["graph_construction"]["time"],
            
            # Path Generation Details
            "paths_generated": result["path_generation"]["count"],
            "selected_path_index": result["path_generation"].get("selected_path_index", -1),
            
            # Infilling Details
            "infilling_skipped": result["infilling"].get("skipped", False),
            "infilling_time": result["infilling"].get("time", 0) if not result["infilling"].get("skipped", True) else 0,
            "infilling_success": result["infilling"].get("success", False),
            
            # Verification Details
            "verification_time": result["verification"].get("time", 0),
            "verification_triples_processed": result["verification"].get("details", {}).get("triples_processed", 0),
            "verification_support_count": result["verification"].get("details", {}).get("support_count", 0),
            "verification_refute_count": result["verification"].get("details", {}).get("refute_count", 0),
            "verification_nei_count": result["verification"].get("details", {}).get("nei_count", 0),
            "verification_final_prediction": result["verification"].get("details", {}).get("final_prediction", ""),
            
            # Overall Metrics
            "total_llm_calls": result["total_llm_calls"],
            "elapsed_time": result["elapsed_time"],
            "error": result["error"] or "",
            
            # Detailed JSON fields for deep analysis
            "graph_details_json": json.dumps(result["graph_construction"]["details"], ensure_ascii=False),
            "path_generation_json": json.dumps(result["path_generation"], ensure_ascii=False),
            "infilling_details_json": json.dumps(result["infilling"], ensure_ascii=False),
            "verification_details_json": json.dumps(result["verification"], ensure_ascii=False),
            "path_results_json": json.dumps([{
                "path_index": p["path_index"],
                "path": p["path"],
                "infilling": p["infilling"],
                "verification": p["verification"],
                "prediction": p["normalized_prediction"]
            } for p in result.get("path_results", [])], ensure_ascii=False)
        }

        return flat_result

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

    parser = argparse.ArgumentParser(description="Run GraphCheck on ExFever dataset with detailed tracking")
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

    args = parser.parse_args()

    output_file = asyncio.run(benchmark(
        data_path=args.data_path,
        db_path=args.db_path,
        output_file=args.output,
        model_name=args.model,
        path_limit=args.path_limit,
        similarity_top_k=args.top_k,
        max_samples=args.max_samples,
        max_concurrent=args.max_concurrent,
    ))

    logger.info("Running evaluation...")
    evaluate_file(output_file)