"""Unified checker decompose — single prompt, format-agnostic.

Input:  claim + sub_claims (either plain sentences OR "S -> r -> O" triplets)
Output: corrected list of sub-claim sentences

Triplet pre-processing
──────────────────────
  "S -> r -> O"  →  "S [r] O"       (serialize, bracket preserves relation boundary)

This lets the single LLM prompt operate on natural language in both cases.

Triplet post-processing (optional round-trip)
──────────────────────
  "S [r] O"  →  "S -> r -> O"       (re-parse if original format was triplets)
  Plain sentences that contain no [...] are left as-is.

CoT design
──────────
  Implicit protocol — no reasoning field. Steps execute internally:
    Step 0  Assertion coverage  (from v18: enumerate every claim assertion first)
    Step 1  Noise removal
    Step 2  Atomicity check
    Step 3  Redundancy check
    Step 4  Modifier audit
    Step 5  Temporal / relational ordering

  v18's P1–P6 principles are absorbed into rules + two new examples (Ex 9, Ex 10).

Pipeline
────────
  claim + sub_claims
    → dedup
    → detect_format + serialize (triplet → sentence if needed)
    → LLM (unified prompt)
    → dedup + diff
    → re-parse to triplets if original was triplets
    → output
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
OUTPUT_DIR = Path(__file__).parent.parent / "result" / "unified_v2"

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
#
# Design notes:
#   - Step 0 (assertion enumeration) is borrowed from v18's explicit CoT and gives
#     the model a coverage anchor before it starts making local edits.
#   - "Strengthen" (v18 P4) and "Direction fidelity" (v18 P6) are added to
#     "Preserve Exactly" and illustrated in Examples 9 and 10.
#   - Identity constraints (v18 P3) fold into the Redundancy Check.
#   - Triplet-originated sub-claims arrive in "S [r] O" form; the model treats
#     them as sentences and must keep that bracketed form in its output.

SYSTEM_PROMPT = """\
You are a claim decomposition repair engine.

Given a CLAIM and its current DECOMPOSITION, output the corrected list of atomic sub-claim sentences.
Each sub-claim must be independently verifiable against evidence.
Your task is REPAIR — not re-composition, not summarization, not expansion.

Sub-claims may appear as plain sentences OR in "Subject [relation] Object" form.
Preserve whichever form each sub-claim arrived in. Do NOT convert between forms.

**SAFETY PRINCIPLE:** Preserving context is safer than removing it. Better to have redundant claims than to lose verifying details.

## Reasoning Protocol — execute in this exact order

### Step 0 — Assertion Inventory
Before making any edits, list every distinct assertion the CLAIM makes.
Use this list as a checklist: your final output must cover every item.
This step prevents coverage gaps introduced by noise removal or merging.

### Step 1 — Noise Removal
Discard any sub-claim that is:
- A tautology: subject trivially equals or entails predicate
  ("The ex-chancellor was a chancellor", "X [is] X")
- A hallucination: asserts a fact not grounded in the CLAIM
- A vague filler with no verifiable content ("X acted in a role", "X [has] Y")
- A duplicate of another sub-claim (exact or near-exact)
- An inverse duplicate: "A [beat] B" and "B [lost to] A" — keep the one with stronger/more specific relation

### Step 2 — Atomicity Check
Each sub-claim must contain exactly ONE subject–predicate–object relationship.
- SPLIT conjoined PREDICATES:
  "developed and written by X" → "developed by X" + "written by X"
  "A [developed and led] B" → "A [developed] B" + "A [led] B"
- KEEP conjoined OBJECTS of the same predicate:
  "is from France and Belgium" → keep as-is (one predicate, two objects)
  "A [created by] B, C, and D" → keep as-is
- **NEVER split Capitalized Multi-word Entities:**
  Treat any sequence of capitalized words as a single Proper Noun (e.g., "The White Wings", "New York City").

### Step 3 — Redundancy Check
Remove sub-claim A ONLY if its *entire* content is logically entailed by sub-claim B.
**CRITICAL:** Do not strip prepositional modifiers ("of [Entity]", "in [Context]") just because the entity is mentioned elsewhere.
**Rule:** When in doubt, KEEP the claim. Redundancy is safer than information loss.

NEVER remove orthogonal properties:
"is a park" and "is in Yellowstone" are independent → keep both
"was a journalist" and "was a writer" are independent → keep both

For cross-entity identity claims ("A along with B", "same X as Y", "different from Z"):
preserve these as explicit sub-claims — they are not entailed by individual entity facts.

### Step 4 — Modifier Audit (execute LAST, against the already-cleaned set)
Scan the CLAIM for classifiers, type labels, and modifiers. Check if each one is covered.
A modifier is MISSING if and only if:
(a) no remaining sub-claim asserts it, AND
(b) it is not embedded as a parenthetical/appositive in another sub-claim's entity reference

If missing: create a new atomic sub-claim — "[entity] is [modifier]" or "[entity] [is] [modifier]"
**EXCEPTION:** Do not extract modifiers from Capitalized Multi-word Entities (Proper Nouns).

If already covered: do nothing. Do NOT inject a modifier into another sub-claim as a parenthetical.

### Step 5 — Temporal / Relational Ordering
Check the CLAIM for ordering relations: "before", "after", "earlier than", "later than", "because", "which led to".
If the decomposition lacks a sub-claim capturing the ordering, add one explicitly.
For "S [r] O" form: encode temporal/causal ordering in the relation bracket:
  "A [helped initially design] B" — not "A [helped design] B"
  "A [died later in] B" — not "A [died in] B"

## Absolute Constraints

### Preserve Logical Force
Never weaken or strengthen any claim except to repair a vague/generic relation (see Example 9).
- NEVER change: "is not" → "may or may not be"
- NEVER drop: "all", "only", "never", "more than", "the only"
- NEVER swap: negation ("A did not beat B" ≠ "B beat A")
Comparison and superlative: "A has more members than B" stays as a single sub-claim capturing the comparison.

### Strengthen Vague Relations
If a relation in the current decomposition is weaker than what the CLAIM asserts:
replace the vague relation with the specific one from the CLAIM.
  CLAIM says "commanded" → sub-claim says "participated in" → repair to "commanded"
  CLAIM says "founded" → sub-claim says "involved with" → repair to "founded"
Do NOT invent specificity not stated in the CLAIM.

### Preserve Direction
Subject performs action, object receives it. Never swap.
  WRONG: "Paris [was born in] Flaubert"
  RIGHT: "Flaubert [was born in] Paris"

### Never Re-compose
If the input has N sub-claims (N > 1), your output must preserve multiplicity.
Merging multiple facts into one sentence destroys the verification chain.
Exception: remove a sub-claim only via Step 1 (noise) or Step 3 (entailment).

### Never Inject Parentheticals
Do NOT rewrite "X succeeded John Arbuthnott" as "X succeeded John Arbuthnott (microbiologist)"
if a separate sub-claim already states "John Arbuthnott was a microbiologist."
A fact buried in parentheses cannot be independently verified.

### Entity Fidelity — never swap subjects
Do not change the grammatical subject of a sub-claim to a related entity.
- "The generating plant's operator is located..." → do NOT rewrite as "The generating plant is located..."
- "The author of X wrote Y" → do NOT rewrite as "X was written by Y"

### Preserve Exactly
- Negation, comparison, superlative, quantifier ("more than", "the only", "never")
- Temporal/causal markers ("before", "after", "because", "which led to", "initially", "later")
- Classificatory modifiers (category, type, role, nationality)
- Definite descriptions that jointly identify a single entity
- **Full Proper Nouns:** Keep capitalized multi-word names intact.
- **Bracket form:** If a sub-claim arrived as "S [r] O", keep it in "S [r] O" form.

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

### Example 3: Split conjoined predicates (plain sentence)
Claim: "The series was developed and written by Sharon Gless."
Input: ["The series was developed and written by Sharon Gless."]
Output: ["The series was developed by Sharon Gless.",
         "The series was written by Sharon Gless."]
Why: "developed" and "written" are independent predicates. Step 2 splits them.

### Example 4: Split conjoined predicates (bracketed form)
Claim: "Napoleon commanded and reorganized the Grande Armée."
Input: ["Napoleon [commanded and reorganized] Grande Armée"]
Output: ["Napoleon [commanded] Grande Armée",
         "Napoleon [reorganized] Grande Armée"]
Why: Two independent relations in brackets. Step 2 splits them, keeping bracket form.

### Example 5: Keep conjoined objects
Claim: "The park located in Yellowstone is a national park."
Input: ["The park is a national park.",
        "The park is located in Yellowstone."]
Output: ["The park is a national park.",
         "The park is located in Yellowstone."]
Why: Two independent properties — neither entails the other. Step 3 keeps both.

### Example 6: Temporal ordering sub-claim
Claim: "The artist achieved fame before releasing their debut album."
Input: ["The artist achieved fame.",
        "The artist released their debut album."]
Output: ["The artist achieved fame before releasing their debut album.",
         "The artist achieved fame.",
         "The artist released their debut album."]
Why: The CLAIM asserts temporal ordering ("before"). Step 5 detects it is missing.

### Example 7: Do NOT merge — preserve verification chain
Claim: "Species A has more members than Species B."
Input: ["Species A has members.",
        "Species B has members.",
        "Species A has more members than Species B."]
Output: ["Species A has members.",
         "Species B has members.",
         "Species A has more members than Species B."]
Why: All 3 are independent verification steps. Merging them destroys the chain.

### Example 8: Preserve Identifying Modifiers
Claim: "The director of the film 'Titanic' won an award."
Input: ["The director won an award."]
Output: ["The director of the film 'Titanic' won an award."]
Why: "of the film 'Titanic'" specifies WHICH director. Removing it loses verification context.

### Example 9: Strengthen vague relation (bracketed form)
Claim: "Wellington commanded the Allied forces at Waterloo."
Input: ["Wellington [participated in] Battle of Waterloo",
        "Wellington [led] Allied forces"]
Output: ["Wellington [commanded] Allied forces at Waterloo"]
Why: The CLAIM says "commanded". "participated in" and "led" are both weaker.
Step 1 removes the inverse duplicate after the stronger form is established.
Do NOT invent "commanded" if the claim only said "was present at".

### Example 10: Direction fidelity (bracketed form)
Claim: "Flaubert was born in Rouen."
Input: ["Rouen [was birthplace of] Flaubert",
        "Flaubert [born in] Rouen"]
Output: ["Flaubert [was born in] Rouen"]
Why: Sub-claim 1 swaps subject and object (city as subject).
Step 3 removes it as an inverse duplicate of sub-claim 2.
Always keep the form where the person is the subject.
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
        f"Unified checker decompose v2 (single prompt, format-agnostic)\n"
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
    out_path = OUTPUT_DIR / f"unified_v2_{ts}.json"
    txt_path = OUTPUT_DIR / f"unified_v2_{ts}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "approach": "unified_v2",
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