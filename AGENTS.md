# AGENTS.md - AI Coding Agent Guidelines

## Project Overview
Python project for fact-checking/claim verification using LLMs and the `workflows` library. Uses `uv` for dependency management.

## Build, Lint, Test Commands

### Setup
```bash
uv sync                          # Install dependencies
```

### Run Main Workflow (Single Execution)
```bash
uv run python benchmark.py       # Run ViFactCheck benchmark
```

### Run Helper Scripts
```bash
uv run python show_evidence.py   # Display Feverous evidence rendering
uv run python scripts/graph_check/build_index.py  # Build graph index
```

### Run Tests
```bash
# Tests are in tests/graphcheck/ - run individually:
uv run python tests/graphcheck/graphcheck.py
uv run python tests/graphcheck/dp_graphcheck.py
uv run python tests/graphcheck/direct.py
```

## Code Style Guidelines

### Imports
- Standard library first, then third-party, then local imports
- Use absolute imports: `from src.modules.datasets.base import Dataset`
- Group imports logically with blank lines between groups

### Typing
- Use type hints for function parameters and return values
- Use `Optional[T]` for nullable parameters
- Use `list[str]`, `dict` for collections

### Naming Conventions
- Classes: `PascalCase` (e.g., `SimpleBaseFactCheck`, `FactCheckStartEvent`)
- Functions/Variables: `snake_case` (e.g., `fact_check`, `context`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `LABELS`, `SIMPLE_USER`)

### Async/Await
- Use `async def` for I/O-bound operations (LLM calls, DB queries)
- Use `await` when calling async functions
- Wrap async calls in `asyncio.run()` in entry points

### Error Handling
- Use `try/except` blocks for external API calls (OpenAI, DB)
- Validate inputs before processing
- Use meaningful error messages

### Class Design
- Inherit from base classes: `Workflow`, `Dataset`, `StartEvent`, `StopEvent`
- Mark workflow steps with `@step` decorator
- Initialize dependencies in `__init__`

### Dependencies
- `workflows` library: Event-driven workflow orchestration
- `llama_index`: LLM integration (OpenAI)
- `datasets`, `jsonlines`: Data loading
- `pandas`: Data manipulation
- `sklearn`: Evaluation metrics

## Workflow Pattern
```python
from workflows import Workflow, step

class MyWorkflow(Workflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @step
    async def my_step(self, event: MyEvent) -> NextEvent:
        # Process event
        return NextEvent(result)
```

## Dataset Pattern
```python
class MyDataset(Dataset):
    @classmethod
    def from_csv(cls, path: str) -> "MyDataset":
        # Load from CSV
        pass
    
    def __getitem__(self, index: int) -> dict:
        return {"context": ..., "claim": ..., "label": ...}
```

## Key Directories
- `src/impls/` - Implementation classes (events, workflows)
- `src/modules/` - Reusable modules (datasets, prompts, evaluators)
- `tests/graphcheck/` - Test suite
- `scripts/` - Utility scripts

## Known Patterns
- Labels: `SUPPORT`, `REFUTE`, `NEI` (Not Enough Information)
- Prompts use `SIMPLE_USER` or `SIMPLE_REASONING_USER` templates
- Results saved to `result/*.csv`
- LLM responses parsed from `Yes/No/Not Enough Information` → `SUPPORT/REFUTE/NEI`

## Writing Style Guidelines

### Code Comments
- **Be explicit**: Explain *why*, not *what*. The code itself shows *what*.
- **Use complete sentences** with proper grammar and punctuation.
- **Flag trade-offs**: When making non-obvious decisions, document the reasoning.
- **Example**:
  ```python
  # BAD: Cache result to speed up
  cached_result = cache_data(data)  # Cache result to speed up
  
  # GOOD: Cache to avoid recomputing expensive graph traversal
  cached_result = cache_data(data)
  ```

### Documentation
- **Docstrings**: Use `"""` triple quotes. Include:
  - Brief summary (1 sentence)
  - Args (with types)
  - Returns (with type)
  - Raises (if applicable)
- **Example**:
  ```python
  def load_dataset(path: str) -> Dataset:
      """Load dataset from CSV or JSONL file.
      
      Args:
          path: File path to dataset.
          
      Returns:
          Dataset object with data samples.
          
      Raises:
          FileNotFoundError: If path does not exist.
      """
  ```

### Function Naming
- **Verbs for actions**: `load_dataset`, `parse_claim`, `evaluate_result`
- **Avoid vague names**: Use `fetch` instead of `get`, `compute` instead of `process`
- **Boolean functions**: Prefix with `is_`, `has_`, `should_`, `can_`

### Error Messages
- **Be actionable**: Tell user what went wrong AND how to fix it.
- **Example**:
  ```python
  raise ValueError("OPENAI_API_KEY not found. Set it in .env file or as environment variable.")
  ```

## Analysis & Suggestions Guidelines

### When Reviewing Code
1. **Check for existing patterns first**: Does this already exist in the codebase?
2. **Verify type safety**: No `Any`, no `@ts-ignore`, no suppressed type errors.
3. **Confirm tests exist**: New functionality needs tests.
4. **Validate documentation**: Docstrings match implementation.

### Common Issues to Flag
- **Missing error handling** for external APIs (OpenAI, DB queries)
- **Hardcoded values** (paths, API keys, configuration)
- **Unnecessary complexity**: Can this be simplified?
- **Inconsistent patterns**: Does this deviate from existing conventions?

### Suggestion Format
When proposing changes:
1. **State the problem** clearly and concisely.
2. **Explain the impact** (performance, maintainability, bugs).
3. **Provide the fix** (code snippet or specific change).
4. **Ask for confirmation** before implementing.

**Example**:
> I notice you're using `input()` for API key configuration. This exposes credentials in history.
> 
> **Impact**: Security risk, credentials in plain text.
> 
> **Fix**: Use `.env` file with `load_dotenv()` or environment variables.
> 
> Should I proceed with this change?

## Critical Rules (NEVER Violate)

### Type Safety
- **NO** `Any` types without strong justification
- **NO** `@ts-ignore`, `@ts-expect-error`, or suppressed warnings
- **NO** silent type conversion that loses information

### Error Handling
- **NO** empty `except` blocks
- **NO** swallowing exceptions without logging or re-raising
- **ALWAYS** validate external inputs before processing

### Security
- **NEVER** commit API keys, passwords, or secrets
- **NEVER** log sensitive data (tokens, PII, credentials)
- **ALWAYS** use environment variables for configuration

### Testing
- **NEVER** delete failing tests to "pass"
- **ALWAYS** add tests for new functionality
- **ALWAYS** verify tests pass before marking task complete

### Code Quality
- **NEVER** leave code in broken state after failed fixes
- **ALWAYS** verify changes with diagnostics/build/tests
- **ALWAYS** match existing patterns unless there's a strong reason to change

### Project-Specific Rules
- **Use `uv` as the standard Python package manager**: Always use `uv sync`, `uv run`, `uv add`, `uv remove` for dependency management. Never use `pip` or `poetry` unless explicitly instructed.
- **Always present a plan first**: Before implementing any multi-step task, create a detailed work plan and ask for user approval. Do NOT start implementation without explicit confirmation.
- **Always ask before installing dependencies**: Never add new packages with `uv add` without first asking the user and explaining why the package is needed.
