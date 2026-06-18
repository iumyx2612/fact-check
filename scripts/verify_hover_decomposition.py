"""
Verify whether each sub-claim decomposition of a HoVer claim is correct.

Uses llama-index structured_predict with Pydantic for reliable schema-validated output.

Usage:
    uv run python tests/verify_hover_decomposition.py [--limit N] [--output PATH]

Env vars (same as the rest of the project):
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, etc.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike


def create_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: float = 120.0,
    max_retries: int = 3,
    reuse_client: bool = False,
) -> LLM:
    provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()

    if provider == "openai":
        name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        seed = int(os.environ.get("OPENAI_SEED", "42"))
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        return OpenAI(
            model=name,
            api_key=key,
            max_tokens=max_tokens,
            temperature=temperature,
            additional_kwargs={"top_p": top_p, "seed": seed},
            max_retries=max_retries,
            timeout=timeout,
            reuse_client=reuse_client,
        )

    if provider in ("vllm", "custom"):
        name = model_name or os.environ.get("LLM_MODEL_NAME", "Nemotron-3-Super-120B-A12B")
        base_url = api_base or os.environ.get("LLM_API_BASE_URL", "http://localhost:8000/v1")
        key = api_key or os.environ.get("LLM_API_KEY", "not-needed-for-vllm")
        return OpenAILike(
            model=name,
            api_key=key,
            api_base=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            additional_kwargs={"top_p": top_p},
            max_retries=max_retries,
            timeout=timeout,
            reuse_client=reuse_client,
            is_chat_model=True,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}. Supported providers: openai, vllm, custom"
    )

INPUT_JSON = Path("datas/hover/hover_dev_decomposed_v2.json")
DEFAULT_OUTPUT = Path("datas/hover/hover_dev_decomposed_verified.json")


# ---------- Pydantic output schema ----------

class SubClaimVerdict(BaseModel):
    index: int = Field(description="The sub-claim index (matches the numbered list)")
    correct: bool = Field(description="Whether the sub-claim correctly represents the original claim")
    reason: str = Field(description="Short explanation for the verdict")
    correct_sub_claim: str | None = Field(
        description="The corrected triplet (Subject -> Relation -> Object) only if incorrect, otherwise null"
    )


class DecompositionVerdict(BaseModel):
    verdicts: list[SubClaimVerdict] = Field(
        description="One verdict per sub-claim, in the same order as the input list"
    )


# ---------- Prompt ----------

SYSTEM_PROMPT = """\
You are an expert fact decomposition verifier.
You will be given an original claim and a numbered list of sub-claims extracted from it.
Each sub-claim is a triplet: Subject -> Relation -> Object.

Verify whether each sub-claim faithfully and correctly represents the original claim.
Pay close attention to relation direction — a reversed triplet is WRONG.

For each sub-claim provide:
- correct: true/false
- reason: short explanation
- correct_sub_claim: null if correct, or the fixed triplet if incorrect

Example:
  Claim: "Alice is the mother of Bob."
  CORRECT: Alice -> mother of -> Bob  →  correct=true, correct_sub_claim=null
  WRONG:   Bob -> mother of -> Alice  →  correct=false, correct_sub_claim="Alice -> mother of -> Bob"\
"""

USER_TEMPLATE = """\
Original claim:
{claim}

Sub-claims to verify:
{sub_claims_text}
"""


def build_user_prompt(claim: str, sub_claims: list[dict]) -> str:
    lines = "\n".join(f"{sc['index']}. {sc['sub_claim']}" for sc in sub_claims)
    return USER_TEMPLATE.format(claim=claim, sub_claims_text=lines)


# ---------- OpenAI strict JSON schema ----------

def _strict_schema() -> dict:
    """Build an OpenAI-strict-compatible JSON schema for DecompositionVerdict."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "correct", "reason", "correct_sub_claim"],
                    "properties": {
                        "index": {"type": "integer"},
                        "correct": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "correct_sub_claim": {"type": ["string", "null"]},
                    },
                },
            }
        },
    }


# ---------- Core logic ----------

async def verify_one(uid: str, rec: dict, llm, semaphore: asyncio.Semaphore, idx: int, total: int) -> tuple[str, dict]:
    from llama_index.core.llms import ChatMessage, MessageRole

    claim = rec["claim"]
    sub_claims = rec["sub_claims"]
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=build_user_prompt(claim, sub_claims)),
    ]

    verdict_map: dict[int, SubClaimVerdict] = {}
    async with semaphore:
        try:
            response = await llm.achat(
                messages,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "DecompositionVerdict",
                    "strict": True,
                    "schema": _strict_schema(),
                }},
            )
            raw = response.message.content or "{}"
            result = DecompositionVerdict.model_validate_json(raw)
            verdict_map: dict[int, SubClaimVerdict] = {v.index: v for v in result.verdicts}
        except Exception as e:
            print(f"  [ERROR] uid={uid}: {e}")

    verified_sub_claims = []
    for sc in sub_claims:
        verdict = verdict_map.get(sc["index"])
        verified_sub_claims.append({
            **sc,
            "correct": verdict.correct if verdict else None,
            "reason": verdict.reason if verdict else None,
            "correct_sub_claim": verdict.correct_sub_claim if verdict else None,
        })

    all_correct = all(sc["correct"] is True for sc in verified_sub_claims)
    correct_count = sum(1 for sc in verified_sub_claims if sc.get("correct") is True)
    print(f"[{idx}/{total}] uid={uid[:8]}... {correct_count}/{len(verified_sub_claims)} sub-claims correct")

    return uid, {**rec, "all_correct": all_correct, "sub_claims": verified_sub_claims}


async def verify_all_async(records: dict, llm, limit: int | None = None, concurrency: int = 5) -> dict:
    uids = list(records.keys())
    if limit:
        uids = uids[:limit]

    semaphore = asyncio.Semaphore(concurrency)
    total = len(uids)

    tasks = [
        verify_one(uid, records[uid], llm, semaphore, i + 1, total)
        for i, uid in enumerate(uids)
    ]
    pairs = await asyncio.gather(*tasks)
    return dict(pairs)


def main():
    parser = argparse.ArgumentParser(description="Verify HoVer claim decompositions with LLM")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N uids")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel requests")
    args = parser.parse_args()

    print(f"Loading {INPUT_JSON} ...")
    with open(INPUT_JSON, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records.")

    llm = create_llm(max_tokens=2048, temperature=0.0)
    print(f"LLM ready: {llm.__class__.__name__} (concurrency={args.concurrency})")

    results = asyncio.run(verify_all_async(records, llm, limit=args.limit, concurrency=args.concurrency))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_sc = sum(len(r["sub_claims"]) for r in results.values())
    correct_sc = sum(
        sum(1 for sc in r["sub_claims"] if sc.get("correct") is True)
        for r in results.values()
    )
    print(f"\nDone. {correct_sc}/{total_sc} sub-claims correct across {len(results)} claims.")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
