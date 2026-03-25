"""
Run the full ExFever GraphCheck pipeline for a single claim (construct → paths → infill → verify).

Edit the variables in CONFIG below, then:
  cd /path/to/fact-check && uv run python run_graphcheck_claim.py

Requires OPENAI_API_KEY, Milvus (MILVUS_URI / MILVUS_TOKEN), and an indexed collection
(see scripts/graph_check/build_exfever_index.py).

Chi tiết pipeline (construct, graph, infill, verify) được in ra stdout qua StdoutGraphCheckTrace,
không dùng logging DEBUG. Dòng HTTP từ httpx/openai vẫn có thể xuất hiện ở mức INFO.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

from benchmark_exfever_graphcheck import run_graphcheck_for_sample
from src.impls.workflows.graph_check.trace_sink import StdoutGraphCheckTrace
from src.utils.graphcheck_exfever import (
    DEFAULT_PATH_LIMIT,
    DEFAULT_SIMILARITY_TOP_K,
    milvus_retriever_for_graphcheck,
    create_llm,
)

# ---------------------------------------------------------------------------
# CONFIG — sửa trực tiếp tại đây
# ---------------------------------------------------------------------------

CLAIM = """See You Again was remixed by Rock Mafia as the second single from Miley Cyrus's second studio album."""

MODEL = os.environ.get("OPENAI_MODEL", "Nemotron-3-Super-120B-A12B")
PATH_LIMIT = DEFAULT_PATH_LIMIT
SIMILARITY_TOP_K = DEFAULT_SIMILARITY_TOP_K
MAX_TOKENS = 8096
# Ghi JSON kết quả ra file (None = chỉ in ra stdout)
OUTPUT_JSON: str | None = None

# ---------------------------------------------------------------------------
# Logging: không dùng DEBUG cho graph; giữ INFO để thấy request HTTP (httpx).
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)

for _noisy in (
    "llama_index",
    "workflows",
    "pymilvus",
    "asyncio",
    "urllib3",
    "requests",
    "aiohttp",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.INFO)


async def _run(claim: str, model_name: str, path_limit: int, similarity_top_k: int) -> dict:
    llm = create_llm(provider="vllm", model_name=model_name, max_tokens=MAX_TOKENS)
    retriever = milvus_retriever_for_graphcheck(similarity_top_k=similarity_top_k)

    return await run_graphcheck_for_sample(
        llm=llm,
        claim=claim,
        retriever=retriever,
        path_limit=path_limit,
        semaphore=None,
        graph_trace=StdoutGraphCheckTrace(),
    )


def main() -> None:
    claim = CLAIM.strip()
    if not claim:
        raise SystemExit("Đặt CLAIM trong run_graphcheck_claim.py (phần CONFIG) rồi chạy lại.")

    result = asyncio.run(
        _run(
            claim=claim,
            model_name=MODEL,
            path_limit=PATH_LIMIT,
            similarity_top_k=SIMILARITY_TOP_K,
        )
    )

    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    print(text)

    if OUTPUT_JSON:
        os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote result to {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
