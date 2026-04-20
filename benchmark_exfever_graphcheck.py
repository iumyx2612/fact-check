"""
ExFever GraphCheck Benchmark

This script runs the full GraphCheck flow for ExFever dataset:
1. Graph Construction - decompose claim into latent entities and triples
2. Path Generation - generate valid orderings for entity resolution
3. Infilling - resolve latent entities using retrieval + LLM
4. Verification - verify each triple against evidence

Run with:
    cd /path/to/fact-check && uv run python benchmark_exfever_graphcheck.py

Requires ``OPENAI_API_KEY`` and a Milvus collection populated via
``scripts/graph_check/build_exfever_index.py``. Configure ``MILVUS_URI``,
``MILVUS_TOKEN``, and optionally ``OPENAI_MODEL`` in ``.env`` (see ``.env.example``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import List

import pandas as pd
from dotenv import load_dotenv
from llama_index.core.llms import LLM
from tqdm.asyncio import tqdm as tqdm_async

from src.impls.events.graph_check.construct_graph import ConstructGraphStartEvent
from src.impls.events.graph_check.infilling import InfillingStartEvent
from src.impls.events.graph_check.verification import VerificationStartEvent
from src.impls.workflows.graph_check.construct_graph import GraphConstructWorkflow
from src.impls.workflows.graph_check.infilling import InfillingWorkflow
from src.impls.workflows.graph_check.trace_sink import GraphCheckTraceSink, NullGraphCheckTrace
from src.impls.workflows.graph_check.verification import VerificationWorkflow
from src.modules.datasets.exfever import ExFever
from src.modules.evaluator import evaluate_file
from src.utils.graphcheck_exfever import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_PATH_LIMIT,
    DEFAULT_SIMILARITY_TOP_K,
    milvus_retriever_for_graphcheck,
    openai_llm_for_graphcheck,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

load_dotenv(".env")

EXFEVER_DATA_PATH = "datas/ex-fever/dev.csv"
EXFEVER_DB_PATHS = [
    "datas/ex-fever/wiki_db.db",
    "datas/ex-fever/wiki_wo_links.db",
]
OUTPUT_FILE = "result/exfever-graphcheck.csv"


def normalize_prediction(prediction: str) -> str:
    """Normalize prediction to match dataset label format."""
    prediction_mapping = {
        "SUPPORTED": "SUPPORT",
        "NOT_SUPPORTED": "REFUTE",
        "NOT_ENOUGH_INFORMATION": "NEI",
    }
    return prediction_mapping.get(prediction, prediction)


def serialize_verification_results(verification_results: list) -> str:
    """Serialize verification results for CSV storage."""
    if not verification_results:
        return "[]"

    simplified_results = []
    for result in verification_results:
        simplified_results.append(
            {
                "triple": result.get("triple", ""),
                "prediction": result.get("prediction", ""),
                "evidence_length": len(result.get("retrieved_evidence", "")),
            }
        )

    return json.dumps(simplified_results)


def extract_graph_info(graph_obj) -> dict:
    """Extract key information from graph object for CSV storage."""
    if not graph_obj:
        return {}

    return {
        "num_latent_entities": getattr(graph_obj, "num_la_ent", 0),
        "num_triples": len(getattr(graph_obj, "total_triples", [])),
        "has_latent_entities": getattr(graph_obj, "num_la_ent", 0) > 0,
    }


async def run_graphcheck_for_sample(
    llm: LLM,
    claim: str,
    retriever,
    path_limit: int = DEFAULT_PATH_LIMIT,
    semaphore: asyncio.Semaphore | None = None,
    graph_trace: GraphCheckTraceSink | None = None,
) -> dict:
    """Run full GraphCheck flow for a single claim."""
    trace = graph_trace or NullGraphCheckTrace()
    start_time = time.time()

    async def _once() -> dict:
        return await _run_graphcheck_for_sample_impl(
            llm, claim, retriever, path_limit, start_time, trace
        )

    if semaphore:
        async with semaphore:
            return await _once()
    return await _once()


async def _run_graphcheck_for_sample_impl(
    llm: LLM,
    claim: str,
    retriever,
    path_limit: int,
    start_time: float,
    graph_trace: GraphCheckTraceSink,
) -> dict:
    """Implementation of GraphCheck flow for a single claim."""
    total_llm_calls = 0
    try:
        logger.info("Processing claim: %s...", claim[:80])

        construct_start = time.time()
        construct_wf = GraphConstructWorkflow(llm=llm, trace=graph_trace)
        graph_result = await construct_wf.run(
            start_event=ConstructGraphStartEvent(claim=claim)
        )
        construct_time = time.time() - construct_start
        total_llm_calls += 1

        graph_obj = graph_result.graph if hasattr(graph_result, "graph") else graph_result
        graph_info = extract_graph_info(graph_obj)
        logger.debug(
            "Graph constructed with %s latent entities, %s triples",
            graph_obj.num_la_ent,
            len(graph_obj.total_triples),
        )

        paths_preview: list[list[str]] = []
        if graph_obj.num_la_ent > 0:
            paths_preview = graph_obj.get_valid_paths(path_limit)
        graph_trace.graph_latent_and_paths(
            graph_obj.num_la_ent,
            list(graph_obj.la_ent_list),
            path_limit,
            paths_preview,
        )

        if graph_obj.num_la_ent == 0:
            logger.debug("No latent entities, skipping infilling")
            verification_wf = VerificationWorkflow(
                llm=llm,
                retriever=retriever,
                dataset_type="exfever",
                trace=graph_trace,
            )
            result = await verification_wf.run(
                start_event=VerificationStartEvent(graph=graph_obj, path_index=0)
            )
            total_llm_calls += len(result.verification_results)
            normalized_pred = normalize_prediction(result.prediction)
            elapsed = time.time() - start_time
            logger.info(
                "Result: %s | Triples: %s | LLM calls: %s | Time: %.1fs",
                normalized_pred,
                len(graph_obj.total_triples),
                total_llm_calls,
                elapsed,
            )
            return {
                "prediction": normalized_pred,
                "verification_results": result.verification_results,
                "graph_info": graph_info,
                "construction_time": construct_time,
                "total_llm_calls": total_llm_calls,
                "elapsed_time": elapsed,
                "paths_tried": 0,
                "successful_path_index": -1,
            }

        paths = paths_preview
        logger.debug("[graphcheck] %s path(s) generated", len(paths))

        if not paths:
            normalized_pred = normalize_prediction("NOT_SUPPORTED")
            elapsed = time.time() - start_time
            logger.info(
                "Result: %s | No valid paths (num_la_ent=%s) | LLM calls: %s | Time: %.1fs",
                normalized_pred,
                graph_obj.num_la_ent,
                total_llm_calls,
                elapsed,
            )
            return {
                "prediction": normalized_pred,
                "verification_results": [],
                "graph_info": graph_info,
                "construction_time": construct_time,
                "total_llm_calls": total_llm_calls,
                "elapsed_time": elapsed,
                "paths_tried": 0,
                "successful_path_index": -1,
            }

        for path_idx, path in enumerate(paths):
            logger.debug(
                "[graphcheck] --- path %s/%s: %s ---",
                path_idx + 1,
                len(paths),
                path,
            )
            graph_trace.path_only(path_idx, path)

            infilling_start = time.time()
            infilling_wf = InfillingWorkflow(
                llm=llm,
                retriever=retriever,
                dataset_type="exfever",
                trace=graph_trace,
            )

            infilling_result = await infilling_wf.run(
                start_event=InfillingStartEvent(
                    claim=claim, path=path, graph=graph_obj, path_index=path_idx
                )
            )
            infilling_time = time.time() - infilling_start
            total_llm_calls += len(path)

            infilled_graph = infilling_result.graph

            verification_start = time.time()
            verification_wf = VerificationWorkflow(
                llm=llm,
                retriever=retriever,
                dataset_type="exfever",
                trace=graph_trace,
            )

            result = await verification_wf.run(
                start_event=VerificationStartEvent(graph=infilled_graph, path_index=path_idx)
            )
            verification_time = time.time() - verification_start
            total_llm_calls += len(result.verification_results)

            logger.debug(
                "[graphcheck] path[%s] prediction=%s  infill=%.2fs verify=%.2fs triples_checked=%s",
                path_idx,
                result.prediction,
                infilling_time,
                verification_time,
                len(result.verification_results),
            )

            if result.prediction == "SUPPORTED":
                normalized_pred = normalize_prediction(result.prediction)
                graph_trace.path_prediction(path_idx, result.prediction)
                elapsed = time.time() - start_time
                logger.info(
                    "Result: %s | Paths tried: %s/%s | Triples: %s | LLM calls: %s | Time: %.1fs",
                    normalized_pred,
                    path_idx + 1,
                    len(paths),
                    len(infilled_graph.total_triples),
                    total_llm_calls,
                    elapsed,
                )
                return {
                    "prediction": normalized_pred,
                    "verification_results": result.verification_results,
                    "graph_info": extract_graph_info(infilled_graph),
                    "construction_time": construct_time,
                    "infilling_time": infilling_time,
                    "verification_time": verification_time,
                    "total_llm_calls": total_llm_calls,
                    "elapsed_time": elapsed,
                    "paths_tried": path_idx + 1,
                    "successful_path_index": path_idx,
                    "path_taken": path,
                }

        normalized_pred = normalize_prediction("NOT_SUPPORTED")
        elapsed = time.time() - start_time
        logger.info(
            "Result: %s | All %s paths failed | Triples: %s | LLM calls: %s | Time: %.1fs",
            normalized_pred,
            len(paths),
            len(graph_obj.total_triples),
            total_llm_calls,
            elapsed,
        )
        return {
            "prediction": normalized_pred,
            "verification_results": [],
            "graph_info": graph_info,
            "construction_time": construct_time,
            "total_llm_calls": total_llm_calls,
            "elapsed_time": elapsed,
            "paths_tried": len(paths),
            "successful_path_index": -1,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            "Error processing claim '%s...': %s | Time: %.1fs",
            claim[:50],
            e,
            elapsed,
            exc_info=True,
        )
        return {
            "prediction": "REFUTE",
            "error": str(e),
            "elapsed_time": elapsed,
        }


async def benchmark(
    data_path: str,
    db_path: str | List[str],
    output_file: str,
    model_name: str | None = None,
    path_limit: int = DEFAULT_PATH_LIMIT,
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    max_samples: int | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> str:
    """
    Run GraphCheck benchmark on ExFever dataset with parallel processing.

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
    _ = db_path  # CLI compatibility; retrieval uses Milvus

    logger.info("Loading dataset from %s...", data_path)
    dataset = ExFever.from_csv(data_path)

    dataset_size = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    logger.info(
        "Loaded %s samples, processing %s with max_concurrent=%s...",
        len(dataset),
        dataset_size,
        max_concurrent,
    )

    llm = openai_llm_for_graphcheck(model_name)

    logger.info("Connecting to Milvus-backed ExFever retriever...")
    retriever = milvus_retriever_for_graphcheck(similarity_top_k=similarity_top_k)
    logger.info("Retriever ready")

    semaphore = asyncio.Semaphore(max_concurrent)

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
            semaphore=semaphore,
        )

        prediction = result.get("prediction", "NOT_SUPPORTED")

        return {
            "claim": claim,
            "explanation": explanation,
            "label": label,
            "pred": prediction,
            "is_correct": prediction == label,
            "graph_num_latent_entities": result.get("graph_info", {}).get("num_latent_entities", 0),
            "graph_num_triples": result.get("graph_info", {}).get("num_triples", 0),
            "construction_time": result.get("construction_time", 0),
            "infilling_time": result.get("infilling_time", 0),
            "verification_time": result.get("verification_time", 0),
            "total_llm_calls": result.get("total_llm_calls", 0),
            "elapsed_time": result.get("elapsed_time", 0),
            "paths_tried": result.get("paths_tried", 0),
            "successful_path_index": result.get("successful_path_index", -1),
            "verification_results": serialize_verification_results(result.get("verification_results", [])),
            "path_taken": json.dumps(result.get("path_taken", [])) if result.get("path_taken") else "[]",
            "error": result.get("error", ""),
        }

    bench_start = time.time()
    results = await tqdm_async.gather(
        *[process_sample(i) for i in range(dataset_size)],
        desc="ExFever GraphCheck",
    )
    total_time = time.time() - bench_start

    logger.info(
        "Processed %s samples in %.1fs (%.2f samples/s)",
        dataset_size,
        total_time,
        dataset_size / total_time if total_time > 0 else 0,
    )

    out_df = pd.DataFrame(results)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_file, index=False)
    logger.info("Results saved to %s", output_file)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphCheck on ExFever dataset")
    parser.add_argument("--data_path", type=str, default=EXFEVER_DATA_PATH, help="Path to ExFever CSV file")
    parser.add_argument(
        "--db_path",
        type=str,
        nargs="+",
        default=EXFEVER_DB_PATHS,
        help="Path(s) to ExFever wiki database(s). Ignored; Milvus is used.",
    )
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output CSV file")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model (default: OPENAI_MODEL env or gpt-4.1-mini)",
    )
    parser.add_argument("--path_limit", type=int, default=DEFAULT_PATH_LIMIT, help="Maximum number of paths to try")
    parser.add_argument("--top_k", type=int, default=DEFAULT_SIMILARITY_TOP_K, help="Number of documents to retrieve")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum number of concurrent API requests (rate limiting)",
    )

    args = parser.parse_args()

    output_file = asyncio.run(
        benchmark(
            data_path=args.data_path,
            db_path=args.db_path,
            output_file=args.output,
            model_name=args.model,
            path_limit=args.path_limit,
            similarity_top_k=args.top_k,
            max_samples=args.max_samples,
            max_concurrent=args.max_concurrent,
        )
    )

    logger.info("Running evaluation...")
    evaluate_file(output_file)


if __name__ == "__main__":
    main()
