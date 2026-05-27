"""Unified checker decompose v3.2 — principle-based lean prompt.

Changes from v3.1:
  - Collapsed 5-step protocol → 3 compact rules (Remove / Repair / Must Not Change)
  - Cut examples: 10 → 4 (removed over-specific edge cases)
  - Default: "leave unchanged unless provably broken" (was: "look for things to fix")
  - Added explicit ban on active↔passive reformulation
  - Added explicit ban on stripping qualifiers for independent verifiability
  - Added explicit ban on punctuation/capitalization "fixes"
  - Prompt: 8,919 chars → ~3,500 chars (-60%)

Pipeline:
  claim + sub_claims -> dedup -> detect_format -> serialize (triplet -> S [r] O)
    -> LLM (single call, implicit CoT) -> dedup -> diff -> deserialize -> output
"""

import ast
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.2"

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datas", "merged_data_v14_sampled.csv")
MAX_SAMPLES = 100
OUTPUT_DIR = Path(__file__).parent.parent / "result" / "v3_optimized"

TARGET_UIDS = None

SubClaimFormat = Literal["sentence", "triplet"]


# ─── PYDANTIC ──────────────────────────────────────────────────────────────────

class ReconstructResult(BaseModel):
    final_sub_claims: list[str] = Field(
        description=(
            "Corrected list of atomic sub-claim sentences. "
            "Each must be ONE independently verifiable fact. "
            "Triplet-originated claims stay in 'S [r] O' form. "
            "No tautologies, no duplicates, no noise."
        )
    )


# ─── UNIFIED PROMPT ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a claim decomposition repair engine.

Given a CLAIM and its current DECOMPOSITION, output the corrected list of atomic sub-claim sentences.

**Default: leave every sub-claim exactly as written.** Your only job is to remove provable noise and apply a narrow set of explicit repairs. Do not rephrase, merge, restructure, or "improve" sub-claims. If a sub-claim is acceptable as-is, output it verbatim.

**Never collapse multiple sub-claims into one.** Each atomic sub-claim serves as an independent verification step. "A began operating" + "B began operating" + "A began operating after B" must all remain separate — the temporal claim does NOT replace the individual start-time claims.

## Rule 1 — Remove Provable Noise

Remove a sub-claim ONLY if it matches one of these:
- Tautology: subject trivially entails predicate ("The ex-chancellor was a chancellor", "X [is] X")
- Hallucination: asserts a fact entirely absent from the CLAIM — BUT a sub-claim whose entities AND relation can be traced to content in the CLAIM is NOT a hallucination, even if the wording differs slightly
- Exact duplicate: identical text appears elsewhere
- Near-exact duplicate: same fact in different phrasing ("Alice wrote the song" vs "The song was written by Alice") — keep one, remove the other
- Inverse duplicate: "A [beat] B" and "B [lost to] A" — keep the person-as-subject form
- Vague filler: content-free like "X acted in a role" — BUT a sub-claim containing 2+ proper nouns from the CLAIM is NEVER vague filler

A sub-claim with a relative clause (who/that/which/whose) is NOT a duplicate of its parts — keep it.
When removing duplicates, output count SHOULD decrease. Never invent replacements.

## Rule 2 — Repair Structural Issues

Do NOT change wording unless a specific repair below applies:
- Split conjoined predicates: "developed and written by X" → "developed by X" + "written by X". Keep conjoined objects: "is from France and Belgium" → keep as-is.
- Add missing classifier: if the CLAIM labels an entity (e.g., "French novelist Gustave Flaubert") but NO sub-claim asserts that label, add one atomic sub-claim. Never create "X is X" tautologies.
- Add missing temporal: if the CLAIM asserts ordering ("before", "after", "because") and NO sub-claim captures it, add one sub-claim.
- Replace vague pronouns: if a sub-claim uses "it/their/this/that" referring to another sub-claim's entity, replace the pronoun with the specific noun phrase. Never remove qualifiers from noun phrases to "avoid redundancy" — "the director of the film Titanic" is correct, do not shorten to "the director".

## Rule 3 — What You Must NOT Change

- Predicate/relation: copy verbatim. "produced by" stays "produced by", never "created for" or "made by"
- Negation/comparison/superlative/quantifier: "more than", "the only", "never", "all", "only" — preserve exactly
- Entity references: never strip prepositional modifiers ("of [Entity]", "in [Context]") — they are required for independent verification
- Bracket form: if input is "S [r] O", output stays "S [r] O". Do not convert between sentence and bracket forms
- Grammatical voice: do not change active↔passive. "Scott Buck created X" stays as-is, do not rewrite to "X was created by Scott Buck"
- Capitalization and punctuation: preserve original. Do not "fix" punctuation, casing, or spelling
- Never merge multiple sub-claims into one — it destroys the verification chain
- Never add sub-claims not grounded in the CLAIM

## Examples

### 1: Tautology removal
Claim: "The ex-chancellor of Germany was a lawyer."
Input: ["The ex-chancellor of Germany was a chancellor of Germany.", "The ex-chancellor of Germany was a lawyer."]
Output: ["The ex-chancellor of Germany was a lawyer."]

### 2: Split conjoined predicates
Claim: "The series was developed and written by Sharon Gless."
Input: ["The series was developed and written by Sharon Gless."]
Output: ["The series was developed by Sharon Gless.", "The series was written by Sharon Gless."]

### 3: Missing classifier
Claim: "The French novelist Gustave Flaubert wrote Madame Bovary."
Input: ["Gustave Flaubert wrote Madame Bovary."]
Output: ["Gustave Flaubert is a French novelist.", "Gustave Flaubert wrote Madame Bovary."]

### 4: Near-duplicate removal (count decreases)
Claim: "The song was written by Alice and Bob."
Input: ["The song was written by Alice.", "The song was written by Bob.", "Alice wrote the song."]
Output: ["The song was written by Alice.", "The song was written by Bob."]

### 5: Never merge verification chain
Claim: "The Carondelet Canal began operating after the Miami and Erie Canal."
Input: ["The Carondelet Canal began operating.", "The Miami and Erie Canal began operating.", "The Carondelet Canal began operating after the Miami and Erie Canal."]
Output: ["The Carondelet Canal began operating.", "The Miami and Erie Canal began operating.", "The Carondelet Canal began operating after the Miami and Erie Canal."]
"""

USER_PROMPT = """\
CLAIM:
{claim}

CURRENT DECOMPOSITION:
{sub_claims}

Output (JSON only):
"""


# ─── FORMAT DETECTION ─────────────────────────────────────────────────────────

def detect_format(sub_claims: list[str]) -> SubClaimFormat:
    """Heuristic: >50% contain ' -> ' twice → triplet format."""
    if not sub_claims:
        return "sentence"
    count = sum(1 for sc in sub_claims if sc.count(" -> ") >= 2)
    return "triplet" if count > len(sub_claims) / 2 else "sentence"


# ─── SERIALIZER / RE-PARSER ───────────────────────────────────────────────────

def serialize_triplet(triplet: str) -> str:
    """
    "S -> r -> O"  →  "S [r] O"

    Bracketing the relation keeps a parseable boundary even when subject, relation,
    or object contain spaces or punctuation.
    """
    first = triplet.find(" -> ")
    if first == -1:
        return triplet
    last = triplet.rfind(" -> ")
    if last == first:
        return triplet
    subj = triplet[:first].strip()
    rel = triplet[first + 4:last].strip()
    obj = triplet[last + 4:].strip()
    return f"{subj} [{rel}] {obj}"


def reparse_to_triplet(sentence: str) -> str | None:
    """
    "S [r] O"  →  "S -> r -> O"

    Returns None if the sentence doesn't match the bracketed pattern.
    Used for round-trip conversion back to triplet format.
    """
    m = re.fullmatch(r"(.+?) \[(.+?)\] (.+)", sentence.strip())
    if m:
        return f"{m.group(1)} -> {m.group(2)} -> {m.group(3)}"
    return None


def serialize_sub_claims(sub_claims: list[str], fmt: SubClaimFormat) -> list[str]:
    if fmt == "sentence":
        return sub_claims
    return [serialize_triplet(sc) for sc in sub_claims]


def deserialize_sub_claims(sub_claims: list[str], original_fmt: SubClaimFormat) -> list[str]:
    """Round-trip: if original format was triplets, try to re-parse each output."""
    if original_fmt == "sentence":
        return sub_claims
    result = []
    for sc in sub_claims:
        parsed = reparse_to_triplet(sc)
        result.append(parsed if parsed is not None else sc)
    return result


# ─── SHARED UTILITIES ─────────────────────────────────────────────────────────

def deduplicate_sub_claims(sub_claims: list[str]) -> tuple[list[str], list[str]]:
    seen, unique, dupes = set(), [], []
    for sc in sub_claims:
        (dupes if sc in seen else unique).append(sc)
        seen.add(sc)
    return unique, dupes


def compute_diff(original: list[str], final: list[str]) -> dict:
    orig_set, final_set = set(original), set(final)
    return {
        "added": list(final_set - orig_set),
        "removed": list(orig_set - final_set),
    }


def _token_param(max_val: int) -> dict:
    return {"max_completion_tokens": max_val} if "gpt-5" in MODEL else {"max_tokens": max_val}


# ─── LLM ──────────────────────────────────────────────────────────────────────

def reconstruct(claim: str, sub_claims: list[str]) -> ReconstructResult:
    sc_text = "\n".join(f"{i + 1}. {sc}" for i, sc in enumerate(sub_claims))
    prompt = USER_PROMPT.format(
        claim=claim.replace("{", "{{").replace("}", "}}"),
        sub_claims=sc_text.replace("{", "{{").replace("}", "}}"),
    )
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format=ReconstructResult,
        **_token_param(1024),
    )
    return response.choices[0].message.parsed


# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def run_pipeline(sample: dict, fmt: SubClaimFormat | None = None) -> dict:
    claim = sample["claim"]
    original = sample["sub_claims"][:]

    # STEP 0: DEDUP
    deduped, dupes = deduplicate_sub_claims(original)

    # STEP 1: FORMAT DETECTION
    if fmt is None:
        fmt = detect_format(deduped)

    # STEP 2: SERIALIZE (triplets → "S [r] O" sentences)
    serialized = serialize_sub_claims(deduped, fmt)

    # STEP 3: LLM CALL
    result = reconstruct(claim, serialized)

    # STEP 4: DEDUP OUTPUT
    final_sentences, _ = deduplicate_sub_claims(result.final_sub_claims)

    # STEP 5: DIFF (on serialized form — apples-to-apples comparison)
    diff = compute_diff(serialized, final_sentences)

    # STEP 6: ROUND-TRIP (re-parse to triplets if original was triplets)
    final = deserialize_sub_claims(final_sentences, fmt)

    return {
        "format": fmt,
        "dedup_removed": len(dupes),
        "diff_added": diff["added"],
        "diff_removed": diff["removed"],
        "final_sub_claims": final,
    }


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_samples() -> list[dict]:
    samples = []
    with open(DATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(samples) >= MAX_SAMPLES:
                break
            if TARGET_UIDS is not None:
                uid_prefix = row["uid"][:8]
                if uid_prefix not in [t[:8] for t in TARGET_UIDS]:
                    continue
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


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    header = (
        f"Unified checker decompose v3 (optimized prompt, format-agnostic)\n"
        f"Model: {MODEL} | Prompt: {len(SYSTEM_PROMPT)} chars\n"
        f"{'=' * 80}"
    )
    print(header)

    samples = load_samples()
    print(f"Loaded {len(samples)} samples\n")

    all_results = []

    for i, sample in enumerate(samples):
        print(
            f"[{i + 1}/{len(samples)}] {sample['uid'][:12]} | "
            f"{sample['label']} | hops={sample['num_hops']}"
        )
        print(f"Claim: {sample['claim'][:120]}...")
        print(f"Original ({len(sample['sub_claims'])}):")
        for sc in sample["sub_claims"]:
            print(f"  - {sc}")

        try:
            res = run_pipeline(sample)
            result = {
                "uid": sample["uid"],
                "claim": sample["claim"],
                "label": sample["label"],
                "num_hops": sample["num_hops"],
                "original_sub_claims": sample["sub_claims"],
                "result": res,
            }
            all_results.append(result)

            print(f"\n  [{res['format']}] dedup_removed={res['dedup_removed']}")
            if res["diff_added"]:
                print(f"  added ({len(res['diff_added'])}):")
                for t in res["diff_added"]:
                    print(f"    + {t}")
            if res["diff_removed"]:
                print(f"  removed ({len(res['diff_removed'])}):")
                for t in res["diff_removed"]:
                    print(f"    - {t}")
            print(f"  final ({len(res['final_sub_claims'])}):")
            for sc in res["final_sub_claims"]:
                print(f"    • {sc}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
            all_results.append({
                "uid": sample["uid"],
                "claim": sample["claim"],
                "label": sample["label"],
                "num_hops": sample["num_hops"],
                "original_sub_claims": sample["sub_claims"],
                "result": {"error": str(e)},
            })

        print("\n" + "-" * 80 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"v3_optimized_{ts}.json"
    txt_path = OUTPUT_DIR / f"v3_optimized_{ts}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "approach": "v3_optimized",
                "timestamp": ts,
                "model": MODEL,
                "prompt_len": len(SYSTEM_PROMPT),
                "total": len(samples),
                "results": all_results,
            },
            f, indent=2, ensure_ascii=False,
        )

    txt_lines = [header, f"Loaded {len(samples)} samples\n"]
    for r in all_results:
        uid = r["uid"][:12]
        txt_lines.append(f'[{uid}] {r["label"]} | hops={r["num_hops"]}')
        txt_lines.append(f'Claim: {r["claim"]}')
        txt_lines.append(f'Original ({len(r["original_sub_claims"])}):')
        for s in r["original_sub_claims"]:
            txt_lines.append(f"  - {s}")
        if "error" in r["result"]:
            txt_lines.append(f'ERROR: {r["result"]["error"]}')
        else:
            res = r["result"]
            txt_lines.append(f'Format: {res["format"]} | dedup_removed={res["dedup_removed"]}')
            txt_lines.append(f'Final ({len(res["final_sub_claims"])}):')
            for s in res["final_sub_claims"]:
                txt_lines.append(f"  + {s}")
        txt_lines.append("-" * 80)
        txt_lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    print(f"Saved JSON: {out_path}")
    print(f"Saved TXT:  {txt_path}")


if __name__ == "__main__":
    main()
