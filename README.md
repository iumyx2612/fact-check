# Fact Checking (Claim Verification)

## 1) Fact Checking Task

### Overview
**Fact Checking / Claim Verification**: given a **claim** and its accompanying **context/evidence**, the system predicts a **verdict** for the claim.

### Input → output
- **Input**
  - `claim`: the statement to verify.
  - `context`: the textual context used to decide the claim (often aggregated/normalized from multiple evidence pieces).
  - `evidence` (dataset-dependent): gold/reference evidence (may be plain text, or a list of Wikipedia element IDs in Feverous).
- **Output**
  - `verdict`/`label` in one of 3 classes:
    - `SUPPORT`: the context supports the claim
    - `REFUTE`: the context contradicts/refutes the claim
    - `NEI` (Not Enough Information): not enough information to conclude

In this codebase, the current workflow is **prompt-based classification** (the LLM answers `Yes/No/Not Enough Information`), then the answer is mapped to `SUPPORT/REFUTE/NEI`.

## 2) Code structure (`src/`)

```
src/
  impls/
    events/
      base.py                 # workflow input event (context, claim)
    workflows/
      simple.py               # 2 workflows: simple and reasoning (LLM-based)
  modules/
    datasets/
      base.py                 # Dataset interface + LABELS = [SUPPORT, REFUTE, NEI]
      vifactcheck.py          # ViFactCheck loader (CSV)
      viwikifc.py             # ViWiKiFC loader (CSV)
      feverous/               # Feverous loader (JSONL + Wikipedia DB)
    prompts/
      simple.py               # prompt templates (Yes/No/NEI)
    evaluator.py              # evaluate a prediction CSV with sklearn
```

### High-level benchmark flow
1. Load a dataset → iterate samples shaped like `{context, claim, evidence, label}`.
2. Create `FactCheckStartEvent(context, claim)` and run a workflow.
3. Save predictions to CSV (column `pred`) and run the evaluator (`classification_report` + `confusion_matrix`).

## 3) Datasets configured in code

Datasets live under `src/modules/datasets/`.

### 3.1) ViFactCheck (CSV)
- **Loader**: `src/modules/datasets/vifactcheck.py` (`ViFactCheck.from_csv(path)`).
- **Path used in code**: `datas/vifactcheck/test.csv` (see `benchmark.py`).
- **Expected CSV columns**:
  - `Context`: context text
  - `Statement`: claim
  - `Evidence`: evidence text
  - `labels`: integer label mapped as:
    - `0 -> SUPPORT`
    - `1 -> REFUTE`
    - `2 -> NEI`

### 3.2) ViWiKiFC (CSV)
- **Loader**: `src/modules/datasets/viwikifc.py` (`ViWiKiFC.from_csv(path)`).
- **Expected CSV columns**:
  - `context`
  - `claim`
  - `evidence`
  - `gold_label` mapped as:
    - `Supports -> SUPPORT`
    - `Refutes -> REFUTE`
    - `Not_Enough_Information -> NEI`

### 3.3) Feverous (JSONL + Wikipedia DB)
- **Loader**: `src/modules/datasets/feverous/feverous.py` (`Feverous.from_path(dataset_path, db_path)`).
- **Demo paths in code**: (see `show_evidence.py`)
  - `datas/feverous/feverous_dev_challenges.jsonl`
  - `datas/feverous/feverous_wikiv1.db`
- **Idea**:
  - Each annotation contains `claim`, `label`, and a list of `evidence` (IDs pointing to Wikipedia elements such as sentence/cell/item/title...).
  - The loader queries the DB and “renders” `evidence` and `context` into text strings (lines like `- Evidence ...`, `- Context ...`).

## 4) How to run (using `uv`)

### Install dependencies
```bash
uv sync
```

### Set API key (if using OpenAI)
`benchmark.py` calls `load_dotenv()`. You can create a `.env` file at the repo root, e.g.:
- `OPENAI_API_KEY=...`

### Run ViFactCheck benchmark + simple workflow
```bash
uv run python benchmark.py
```

Notes:
- By default, the script writes to `result/vifactcheck-simple(2).csv`. If `result/` does not exist, create it first:
  - `mkdir -p result`

### See how Feverous evidence/context is rendered
```bash
uv run python show_evidence.py
```

## 5) Key components

- **Prompts**: `src/modules/prompts/simple.py`
  - `SIMPLE_USER`: LLM answers only `Yes/No/Not Enough Information`
  - `SIMPLE_REASONING_USER`: LLM answers using a template with `Reasoning` and `Answer`
- **Workflows**: `src/impls/workflows/simple.py`
  - `SimpleBaseFactCheck`: returns the label
  - `SimpleReasoningFactCheck`: parses both reasoning + label (regex), returns a dict `{label, reasoning}`
- **Evaluator**: `src/modules/evaluator.py`
  - `evaluate_file(csv_path)` expects a CSV with two columns: `label` (ground-truth) and `pred` (prediction)

