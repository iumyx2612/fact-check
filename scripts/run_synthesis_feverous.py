from dotenv import load_dotenv
import pandas as pd
import json
from tqdm.asyncio import tqdm as atqdm
import asyncio
load_dotenv()
from workflows import Context
from llama_index.llms.openai import OpenAI

from src.impls.workflows.synthesis.decompose import DecomposeWorkflow
from src.impls.workflows.synthesis.verify import VerifyWorkflow, VerifyStartEvent
from src.impls.workflows.synthesis.context import SynthesisContext
from src.modules.datasets.feverous import MDFeverous


llm = OpenAI(model='gpt-4.1-mini', temperature=0)
dataset = MDFeverous.from_path(
    "datas/feverous/feverous_dev_challenges.jsonl",
    "datas/feverous/feverous_wikiv1.db"
)
output_path = "result/verify/feverous_simple.csv"

NUM_WORKERS = 4


async def run_decompose(decompose_wf, claim):
    result = await decompose_wf.run(claim=claim)
    return result.sub_claims


async def run_verify(wf, claim, documents, sub_claims, ctx):
    result = await wf.run(start_event=VerifyStartEvent(
        claim=claim,
        documents=documents,
        sub_claims=sub_claims,
    ), ctx=ctx)

    return result.result


async def worker(semaphore, decompose_wf, wf, sample):
    async with semaphore:
        claim = sample.claim
        documents = sample.context
        sample_dict = sample.model_dump()
        try:
            sub_claims = await run_decompose(decompose_wf, claim)

            ctx = Context[SynthesisContext](wf)
            result = await run_verify(wf, claim, documents, sub_claims, ctx)
            ctx_dict = ctx.to_dict()
            state = ctx_dict["state"]["state_data"]
            sub_claim_mapping = json.loads(state)["value"]["verify_ctx"]["verify_mapping"]

            sample_dict["context"] = "\n\n".join(sample_dict["context"])
            sample_dict["pred"] = result
            sample_dict["mapping"] = json.dumps(sub_claim_mapping, ensure_ascii=False)
        except Exception as e:
            print(f"BUG: {e}")
            sample_dict["pred"] = "BUG"
            sample_dict["mapping"] = "BUG"
        return sample_dict


async def async_main():
    decompose_wf = DecomposeWorkflow(llm=llm, timeout=3000)
    wf = VerifyWorkflow(llm=llm, use_reasoning=True, timeout=3000)
    semaphore = asyncio.Semaphore(NUM_WORKERS)

    samples = list(dataset)
    batch_size = 50
    first_batch = True

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]
        tasks = [worker(semaphore, decompose_wf, wf, sample) for sample in batch]
        results = await atqdm.gather(*tasks, desc=f"Batch {batch_start // batch_size + 1}")

        batch_df = pd.DataFrame(results)
        if first_batch:
            batch_df.to_csv(output_path, index=False)
            first_batch = False
        else:
            batch_df.to_csv(output_path, mode='a', header=False, index=False)
        print(f"Saved rows {batch_start} - {batch_start + len(results) - 1}")


if __name__ == '__main__':
    # sync_main()
    asyncio.run(async_main())
