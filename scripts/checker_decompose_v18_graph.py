"""V18: Quality-optimized prompt — condensed principles, merged rules, 150-word CoT cap.

Changes from v17:
  - P1-P6 principles condensed (removed redundant examples)
  - Merged Triplet Structure + Final Output Rules (eliminated overlap)
  - Added 150-word cap on reasoning (preserves quality, cuts verbosity)
  - Same 5-step CoT structure, same direct-output architecture

Pipeline:
  claim + sub_claims -> dedup -> build graph -> reconstruct (LLM, single call, CoT)
    -> final_sub_claims -> client-side diff -> output
"""

import csv
import json
import os
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.2"

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datas", "merged_data_v14_sampled.csv"
)
OUTPUT_DIR = Path(__file__).parent.parent / "result" / "v18_quality_v14"


# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────

class ReconstructResult(BaseModel):
    reasoning: str = Field(
        description="Step-by-step analysis: (1) list every assertion in the claim, "
                    "(2) check coverage, (3) identify gaps, (4) plan fixes, (5) verify."
    )
    final_sub_claims: list[str] = Field(
        description="The COMPLETE corrected list of triplets. Each: 'S -> r -> O'. "
                    "No tautologies, no duplicates, no subsumed triplets."
    )


# ─── PROMPT ───────────────────────────────────────────────────────────────────

RECONSTRUCT_PROMPT = """Reconstruct the graph so it fully and faithfully represents the claim.

CLAIM (THE ONLY SOURCE OF TRUTH):
{claim}

CURRENT GRAPH (DEGRADED — NEEDS REPAIR):
{graph}

The graph shows each entity and its outgoing edges. Entities marked [isolated] have no connections.

YOUR JOB:
Output the COMPLETE corrected list of triplets that fully represents every assertion in the claim.
This is NOT a diff — output the entire final graph.

REASONING (in "reasoning" field, max 150 words):
  Step 1: List EVERY assertion in the claim (number them 1, 2, 3...)
  Step 2: For each assertion, check if the current graph covers it faithfully
  Step 3: Identify specific gaps/errors (missing, weakened, wrong, redundant)
  Step 4: Plan the fix for each gap (add, strengthen, remove, merge)
  Step 5: Verify your final triplet list covers ALL numbered assertions

REPAIR PRINCIPLES:

P1. LOGICAL FORCE — preserve negation, comparison, superlative, quantifier, exclusivity.
    "not both" stays "not both X with". "more X than Y" stays "more X than -> Y".
    NEVER weaken ("is not" to "may or may not", drop "all"/"only"). NEVER introduce falsehoods.

P2. TEMPORAL/CAUSAL — encode "initially", "later", "because", "before/after" in relation.
    "helped initially design", "died later in". Put "because" in RELATION, cause in object.

P3. IDENTITY CONSTRAINTS — "same X as Y", "different from Z", "along with".
    Cross-entity triplets: "A -> chosen along with -> B".

P4. STRENGTHEN — generic to specific. "participated" to "commanded". "included" to "consisted of".

P5. NOISE — remove tautologies ("X -> is -> X"), inverse duplicates, subsumed triplets.

P6. DIRECTION — subject performs action, object receives it. Never swap.

OUTPUT RULES:
- Format: "Subject -> relation -> Object" (noun phrase -> verb phrase -> noun phrase).
- Use EXACT entity names from current graph. Only add new entities if explicitly named in claim.
- Complete final list. Every triplet entailed by claim. Every assertion covered.
- Minimize triplets while maximizing coverage.

OUTPUT (JSON only):
{{"reasoning":"step-by-step analysis","final_sub_claims":["S -> r -> O", ...]}}
"""


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def parse_triplet(triplet: str) -> tuple[str, str, str] | None:
    first = triplet.find(" -> ")
    if first == -1:
        return None
    last = triplet.rfind(" -> ")
    if last == first:
        return None
    return triplet[:first].strip(), triplet[first+4:last].strip(), triplet[last+4:].strip()


def deduplicate_sub_claims(sub_claims: list[str]) -> tuple[list[str], list[str]]:
    seen = set()
    unique = []
    duplicates = []
    for sc in sub_claims:
        if sc in seen:
            duplicates.append(sc)
        else:
            seen.add(sc)
            unique.append(sc)
    return unique, duplicates


def post_process_triplets(triplets: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Reject malformed triplets."""
    valid = []
    rejected = []
    for t in triplets:
        parsed = parse_triplet(t)
        if parsed is None:
            rejected.append((t, "invalid format"))
            continue
        valid.append(t)
    return valid, rejected


def compute_diff(original: list[str], final: list[str]) -> dict:
    """Diff original vs final: added, removed, replaced."""
    orig_set = set(original)
    final_set = set(final)

    added = list(final_set - orig_set)
    removed = list(orig_set - final_set)

    # Detect replacements: removed + added triplet sharing same subject AND same object
    # (relation changed) OR same subject (relation+object changed)
    replaced = []
    added_remain = []
    for a in added:
        a_parsed = parse_triplet(a)
        if a_parsed is None:
            added_remain.append(a)
            continue
        a_subj, a_rel, a_obj = a_parsed
        matched = False
        for r in removed:
            r_parsed = parse_triplet(r)
            if r_parsed is None:
                continue
            r_subj, r_rel, r_obj = r_parsed
            # Same subject + same object = relation changed
            # Same subject + similar object = relation+object changed
            if a_subj == r_subj and (a_obj == r_obj or a_rel == r_rel):
                replaced.append({"old": r, "new": a})
                matched = True
                break
        if not matched:
            added_remain.append(a)

    return {
        "added": added_remain,
        "removed": removed,
        "replaced": replaced,
    }


# ─── GRAPH BUILDER ────────────────────────────────────────────────────────────

def build_graph_for_llm(sub_claims: list[str]) -> str:
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    all_entities: set[str] = set()

    for sc in sub_claims:
        parsed = parse_triplet(sc)
        if parsed is None:
            continue
        subj, rel, obj = parsed
        outgoing[subj].append((rel, obj))
        all_entities.add(subj)
        all_entities.add(obj)

    adj_lines = []
    for entity in sorted(all_entities):
        edges = outgoing.get(entity, [])
        if edges:
            adj_lines.append(f"{entity}")
            for rel, obj in edges:
                adj_lines.append(f"  -> {rel} -> {obj}")
        else:
            adj_lines.append(f"{entity} [isolated]")

    flat_lines = []
    seen = set()
    for subj in sorted(outgoing.keys()):
        for rel, obj in outgoing[subj]:
            edge = f"{subj} -> {rel} -> {obj}"
            if edge not in seen:
                flat_lines.append(edge)
                seen.add(edge)

    return "\n".join(adj_lines) + "\n\nTRIPLET LIST:\n" + "\n".join(flat_lines)


# ─── LLM FUNCTION ─────────────────────────────────────────────────────────────

def _token_param(max_val: int) -> dict:
    if "gpt-5" in MODEL:
        return {"max_completion_tokens": max_val}
    return {"max_tokens": max_val}


def reconstruct(claim: str, graph: str) -> ReconstructResult:
    prompt = RECONSTRUCT_PROMPT.format(
        claim=claim.replace("{", "{{").replace("}", "}}"),
        graph=graph.replace("{", "{{").replace("}", "}}"),
    )
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise graph reconstruction engine. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format=ReconstructResult,
        **_token_param(2048),
    )
    return response.choices[0].message.parsed


# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def run_v18_pipeline(sample: dict) -> dict:
    claim = sample["claim"]
    current_sub_claims = sample["sub_claims"][:]

    # STEP 0: DEDUP
    current_sub_claims, dupes = deduplicate_sub_claims(current_sub_claims)

    # STEP 1: BUILD GRAPH
    graph = build_graph_for_llm(current_sub_claims)

    # STEP 2: SINGLE LLM CALL — CoT + DIRECT FINAL OUTPUT
    result = reconstruct(claim, graph)
    final_raw = result.final_sub_claims
    reasoning = result.reasoning

    # STEP 3: DEDUP + POST-PROCESS
    final_raw, _ = deduplicate_sub_claims(final_raw)
    final, rejected = post_process_triplets(final_raw)

    # STEP 4: CLIENT-SIDE DIFF (audit trail)
    diff = compute_diff(current_sub_claims, final)

    return {
        "dedup_removed": len(dupes),
        "reasoning": reasoning,
        "diff_added": diff["added"],
        "diff_removed": diff["removed"],
        "diff_replaced": diff["replaced"],
        "rejected": [(t, r) for t, r in rejected],
        "final_sub_claims": final,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def load_all_samples() -> list[dict]:
    import ast
    samples = []
    with open(DATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sub_claims = ast.literal_eval(row["sub_claim"])
            except (ValueError, SyntaxError):
                sub_claims = []
            samples.append({
                "uid": row["uid"],
                "claim": row["claim"],
                "label": row["label"],
                "num_hops": row["num_hops"],
                "sub_claims": sub_claims,
            })
    return samples


def main():
    print(f"V18 Quality: Condensed Prompt + 150-word CoT")
    print(f"Model: {MODEL}")
    print()

    samples = load_all_samples()
    print(f"Loaded {len(samples)} samples")
    print()

    all_results = []

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['uid'][:12]} | hops={sample['num_hops']} | {sample['claim'][:120]}")
        try:
            res = run_v18_pipeline(sample)
            result = {
                "uid": sample["uid"],
                "claim": sample["claim"],
                "label": sample["label"],
                "num_hops": sample["num_hops"],
                "original_sub_claims": sample["sub_claims"],
                "result": res,
            }
            all_results.append(result)

            print(f"  dedup_removed={res['dedup_removed']}")
            if res.get("reasoning"):
                print(f"  reasoning: {res['reasoning'][:300]}...")
            if res["diff_replaced"]:
                print(f"  replaced({len(res['diff_replaced'])}):")
                for rp in res["diff_replaced"]:
                    print(f"    ~ {rp['old']} => {rp['new']}")
            if res["diff_added"]:
                print(f"  added({len(res['diff_added'])}):")
                for t in res["diff_added"]:
                    print(f"    + {t}")
            if res["diff_removed"]:
                print(f"  removed({len(res['diff_removed'])}):")
                for t in res["diff_removed"]:
                    print(f"    - {t}")
            if res["rejected"]:
                print(f"  rejected({len(res['rejected'])}):")
                for t, r in res["rejected"]:
                    print(f"    ? {t} ({r})")
            print(f"  final triplets ({len(res['final_sub_claims'])}):")
            for sc in res["final_sub_claims"]:
                print(f"    {sc}")
            print()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}\n")
            all_results.append({
                "uid": sample["uid"],
                "claim": sample["claim"],
                "label": sample["label"],
                "num_hops": sample["num_hops"],
                "original_sub_claims": sample["sub_claims"],
                "result": {"error": str(e)},
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"v18_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "approach": "v18_quality",
            "timestamp": ts,
            "model": MODEL,
            "total": len(samples),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved to: {out_path}")


if __name__ == "__main__":
    main()
