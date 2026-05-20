import csv
import json
import os
import ast
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.2"

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datas", "merged_data_v14_sampled.csv")
OUTPUT_DIR = Path(__file__).parent.parent / "result" / "v31_100sample"

TARGET_UIDS = None


class ReconstructResult(BaseModel):
    final_sub_claims: list[str] = Field(
        description="Corrected list of atomic sub-claim sentences. "
                    "Each must be ONE verifiable fact. "
                    "No tautologies, no duplicates, no noise."
    )


RECONSTRUCT_SYSTEM_V31 = """\
You are a claim decomposition repair engine.

Given a CLAIM and its current DECOMPOSITION, output the corrected list of atomic sub-claim sentences.
Each sub-claim must be independently verifiable against evidence.
Your task is REPAIR — not re-composition, not summarization, not expansion.
**SAFETY PRINCIPLE:** Preserving context is safer than removing it. Better to have redundant claims than to lose verifying details.

## Reasoning Protocol — execute in this exact order

### Step 1 — Noise Removal
Discard any sub-claim that is:
- A tautology: subject trivially equals or entails predicate
  ("The ex-chancellor was a chancellor", "The dog is an animal")
- A hallucination: asserts a fact not grounded in the CLAIM or its decomposition
- A vague filler with no verifiable content ("X acted in a role", "X has a Y")
- A duplicate of another sub-claim (exact or near-exact)

### Step 2 — Atomicity Check
Each sub-claim must contain exactly ONE subject–predicate–object relationship.
- SPLIT conjoined PREDICATES:
  "developed and written by X" → "developed by X" + "written by X"
  "born and raised in Chicago" → "born in Chicago" + "raised in Chicago"
- KEEP conjoined OBJECTS of the same predicate:
  "is from France and Belgium" → keep as-is (one predicate, two objects)
  "created by A, B, and C" → keep as-is
- **NEVER split Capitalized Multi-word Entities:**
  Treat any sequence of capitalized words as a single Proper Noun (e.g., "The White Wings", "New York City"). Do not extract adjectives or nouns from these names.

### Step 3 — Redundancy Check
Remove sub-claim A ONLY if its *entire* content is logically entailed by sub-claim B.
**CRITICAL:** Do not strip prepositional modifiers (e.g., "of [Entity]", "in [Context]") just because the entity is mentioned elsewhere.
- "The remix video of [Film]" provides specific verification context. Do not reduce it to just "The remix video".
- "The director of [Work]" identifies a specific role. Do not reduce it to just "The director".
**Rule:** When in doubt, KEEP the claim. Redundancy is safer than information loss.

NEVER remove orthogonal properties:
"is a park" and "is in Yellowstone" are independent → keep both
"was a journalist" and "was a writer" are independent → keep both

### Step 4 — Modifier Audit (execute LAST, against the already-cleaned set)
Scan the CLAIM for classifiers, type labels, and modifiers. Check if each one is covered.
A modifier is MISSING if and only if:
(a) no remaining sub-claim asserts it, AND
(b) it is not embedded as a parenthetical/appositive in another sub-claim's entity reference

If missing: create a new atomic sub-claim — "[entity] is [modifier]"
**EXCEPTION:** Do not extract modifiers that are part of a Capitalized Multi-word Entity (Proper Noun). Treat the full name as a single atomic unit.

If already covered: do nothing. Do NOT inject a modifier into another sub-claim as a parenthetical.

### Step 5 — Temporal / Relational Ordering
Check the CLAIM for ordering relations: "before", "after", "earlier than", "later than", "compared to".
If the decomposition lacks a sub-claim capturing the ordering, add one that states the ordering explicitly.
CLAIM: "A obtained X before B did Y."
Missing: "A obtained X before B did Y." (or equivalent ordering sub-claim)
→ Add it.

## Absolute Constraints

### Never Re-compose
If the input has N sub-claims (N > 1), your output must preserve multiplicity.
Merging multiple facts into one sentence destroys the verification chain.
Exception: remove a sub-claim only via Step 1 (noise) or Step 3 (entailment).

### Never Inject Parentheticals
Do NOT rewrite "X succeeded John Arbuthnott" as "X succeeded John Arbuthnott (microbiologist)"
if a separate sub-claim already states "John Arbuthnott was a microbiologist."
Parenthetical form hides a fact inside another sub-claim and creates redundancy.
A fact buried in parentheses cannot be independently verified.

### Entity Fidelity — never swap subjects
Do not change the grammatical subject of a sub-claim to a related entity.
- If the original says "The generating plant's operator is located...", do NOT rewrite it as "The generating plant is located..."
- If the original says "The author of X wrote Y", do NOT rewrite it as "X was written by Y"
- Preserve the exact entity being asserted. The subject is the entity to verify.

### Preserve Exactly
- Negation, comparison, superlative, quantifier ("more than", "the only", "never")
- Temporal/causal markers ("before", "after", "because", "which led to")
- Classificatory modifiers (words assigning category, type, role, nationality)
- Definite descriptions that jointly identify a single entity
- Logical force — do not weaken or strengthen any claim
- **Full Proper Nouns:** Keep capitalized multi-word names intact.

## Examples (Synthetic — for illustration only)

### Example 1: Tautology removal
Claim: "The ex-chancellor of Germany was a lawyer."
Input: ["The ex-chancellor of Germany was a chancellor of Germany.",
        "The ex-chancellor of Germany was a lawyer."]
Output: ["The ex-chancellor of Germany was a lawyer."]
Why: Sub-claim 1 is a tautology — "ex-chancellor" trivially entails "chancellor".

### Example 2: Missing classifier extraction
Claim: "The French novelist Gustave Flaubert wrote Madame Bovary."
Input: ["Gustave Flaubert wrote Madame Bovary."]
Output: ["Gustave Flaubert is a French novelist.",
         "Gustave Flaubert wrote Madame Bovary."]
Why: "French novelist" is a classifier present in the CLAIM but missing from decomposition.
Step 4 adds it as a new atomic sub-claim.

### Example 3: Split conjoined predicates
Claim: "The series was developed and written by Sharon Gless."
Input: ["The series was developed and written by Sharon Gless."]
Output: ["The series was developed by Sharon Gless.",
         "The series was written by Sharon Gless."]
Why: "developed" and "written" are independent predicates. Step 2 splits them.

### Example 4: Keep conjoined objects
Claim: "The park located in Yellowstone is a national park."
Input: ["The park is a national park.",
        "The park is located in Yellowstone."]
Output: ["The park is a national park.",
         "The park is located in Yellowstone."]
Why: Two independent properties — neither entails the other. Step 3 keeps both.

### Example 5: Temporal ordering sub-claim
Claim: "The artist achieved fame before releasing their debut album."
Input: ["The artist achieved fame.",
        "The artist released their debut album."]
Output: ["The artist achieved fame before releasing their debut album.",
         "The artist achieved fame.",
         "The artist released their debut album."]
Why: The CLAIM asserts temporal ordering ("before"). Step 5 detects it is missing.
Sub-claim 1 is rewritten to preserve the ordering relation.

### Example 6: Do NOT merge — preserve verification chain
Claim: "Species A has more members than Species B."
Input: ["Species A has members.",
        "Species B has members.",
        "Species A has more members than Species B."]
Output: ["Species A has members.",
         "Species B has members.",
         "Species A has more members than Species B."]
Why: All 3 are independent verification steps. Merging them destroys the chain.
BAD OUTPUT: ["Species A has more members than Species B."]
— collapses 3 steps into 1, violates NEVER re-compose.

### Example 7: Preserve Identifying Modifiers
Claim: "The director of the film 'Titanic' won an award."
Input: ["The director won an award."]
Output: ["The director of the film 'Titanic' won an award."]
Why: "of the film 'Titanic'" is an IDENTIFYING modifier. It specifies WHICH director.
Step 3 keeps it. Removing it would lose verification context.

### Example 8: Never split Proper Nouns
Claim: "The Red Hot Chili Peppers performed at the festival."
Input: ["The Red Hot Chili Peppers performed at the festival."]
Output: ["The Red Hot Chili Peppers performed at the festival."]
Why: "Red Hot Chili Peppers" is a Proper Noun. Step 2 never splits it.
BAD OUTPUT: ["The Chili Peppers performed at the festival.", "The Chili Peppers are red and hot."]
— incorrectly splits a band name.
"""

RECONSTRUCT_USER = """\
CLAIM:
{claim}

CURRENT DECOMPOSITION:
{sub_claims}

Output (JSON only):
"""


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


def compute_diff(original: list[str], final: list[str]) -> dict:
    orig_set = set(original)
    final_set = set(final)
    return {
        "added": list(final_set - orig_set),
        "removed": list(orig_set - final_set),
    }


def _token_param(max_val: int) -> dict:
    if "gpt-5" in MODEL:
        return {"max_completion_tokens": max_val}
    return {"max_tokens": max_val}


def reconstruct(claim: str, sub_claims: list[str]) -> ReconstructResult:
    sc_text = "\n".join(f"{i+1}. {sc}" for i, sc in enumerate(sub_claims))
    prompt = RECONSTRUCT_USER.format(
        claim=claim.replace("{", "{{").replace("}", "}}"),
        sub_claims=sc_text.replace("{", "{{").replace("}", "}}"),
    )

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": RECONSTRUCT_SYSTEM_V31},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format=ReconstructResult,
        **_token_param(1024),
    )
    return response.choices[0].message.parsed


def run_pipeline(sample: dict) -> dict:
    claim = sample["claim"]
    current_sub_claims = sample["sub_claims"][:]

    current_sub_claims, dupes = deduplicate_sub_claims(current_sub_claims)
    result = reconstruct(claim, current_sub_claims)
    final_raw = result.final_sub_claims
    final_raw, _ = deduplicate_sub_claims(final_raw)
    diff = compute_diff(current_sub_claims, final_raw)

    return {
        "dedup_removed": len(dupes),
        "diff_added": diff["added"],
        "diff_removed": diff["removed"],
        "final_sub_claims": final_raw,
    }


def load_samples() -> list[dict]:
    samples = []
    with open(DATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def main():
    header = f"V31 — {'all rows' if not TARGET_UIDS else len(TARGET_UIDS)} from {os.path.basename(DATA_PATH)}\nModel: {MODEL}\nPrompt: {len(RECONSTRUCT_SYSTEM_V31)} chars\n{'=' * 80}"
    print(header)

    samples = load_samples()
    load_msg = f"Loaded {len(samples)} samples\n"
    print(load_msg)

    all_results = []

    for i, sample in enumerate(samples):
        line = f"[{i+1}/{len(samples)}] {sample['uid'][:12]} | {sample['label']} | hops={sample['num_hops']}"
        claim_line = f"Claim: {sample['claim'][:120]}..."
        print(f"{line}\n{claim_line}")
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

            print(f"\n  V31 ({len(res['final_sub_claims'])}) claims:")
            for sc in res["final_sub_claims"]:
                print(f"    • {sc}")
            if res["diff_added"]:
                print(f"  added:")
                for t in res["diff_added"]:
                    print(f"    + {t}")
            if res["diff_removed"]:
                print(f"  removed:")
                for t in res["diff_removed"]:
                    print(f"    - {t}")

        except Exception as e:
            import traceback
            err_out = traceback.format_exc()
            print(err_out)
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
    out_path = OUTPUT_DIR / f"v31_4failed_{ts}.json"
    txt_path = OUTPUT_DIR / f"v31_4failed_{ts}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "approach": "v31",
            "timestamp": ts,
            "model": MODEL,
            "prompt_len": len(RECONSTRUCT_SYSTEM_V31),
            "total": len(samples),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    txt_lines = [header, load_msg]
    for r in all_results:
        uid = r["uid"][:12]
        txt_lines.append(f'[{uid}] {r["label"]} | hops={r["num_hops"]}')
        txt_lines.append(f'Claim: {r["claim"]}')
        txt_lines.append(f'Subclaims ({len(r["original_sub_claims"])}):')
        for s in r["original_sub_claims"]:
            txt_lines.append(f'  - {s}')
        if "error" in r["result"]:
            txt_lines.append(f'ERROR: {r["result"]["error"]}')
        else:
            txt_lines.append(f'Final subclaims ({len(r["result"]["final_sub_claims"])}):')
            for s in r["result"]["final_sub_claims"]:
                txt_lines.append(f'  + {s}')
        txt_lines.append("-" * 80)
        txt_lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    print(f"Saved JSON: {out_path}")
    print(f"Saved TXT:  {txt_path}")


if __name__ == "__main__":
    main()
