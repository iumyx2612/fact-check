from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
import csv
from pathlib import Path

from llama_index.llms.openai import OpenAI

from src.impls.events.base import FactCheckStartEvent
from src.impls.workflows.simple import (
    EvidenceSimpleBaseFactCheck,
    EvidenceSimpleReasoningFactCheck,
)
from src.modules.datasets.feverous import FeverousEvidenceFormat
from src.modules.evaluator import evaluate_file

load_dotenv()


dataset = FeverousEvidenceFormat.from_path("/raid/Workspace/an/code/factcheck/FEVEROUS/data/dev.jsonl",
                             "/raid/Workspace/an/code/factcheck/FEVEROUS/data/feverous_wikiv1.db")
llm = OpenAI(model="gpt-4.1-mini")

# ==========================================
wf = EvidenceSimpleBaseFactCheck(llm=llm)
# ==========================================


async def benchmark(output_file: str):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["context", "claim", "evidence", "label", "pred"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, sample in enumerate(tqdm(dataset)):
            try:
                context = sample["context"]
                claim = sample["claim"]
                evidence = sample["evidence"]
                label = sample["label"]

                start_ev = FactCheckStartEvent(context=evidence, claim=claim)
                output = await wf.run(start_event=start_ev)

                # Workflows may return either a StopEvent-like object with `.result`
                # or the raw result directly (e.g., a dict).
                result = output.result if hasattr(output, "result") else output
                if isinstance(result, dict):
                    prediction = str(result.get("label"))
                else:
                    prediction = str(result)

                writer.writerow(
                    {
                        "context": context,
                        "claim": claim,
                        "evidence": evidence,
                        "label": label,
                        "pred": prediction,
                    }
                )
                f.flush()
            except Exception as e:
                print(f"[benchmark] error at idx={i}: {e}")
                await asyncio.sleep(1)
                continue


if __name__ == '__main__':
    asyncio.run(benchmark("results/feverous-simple-evidence-format.csv"))
    evaluate_file("results/feverous-simple-evidence-format.csv")