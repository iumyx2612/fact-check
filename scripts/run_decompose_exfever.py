from dotenv import load_dotenv
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm as atqdm
load_dotenv()
from llama_index.llms.openai import OpenAI

from src.impls.workflows.synthesis.decompose import DecomposeWorkflow
from src.modules.datasets.ex_fever import MDExFever


llm = OpenAI(model='gpt-4.1-mini', temperature=0)
dataset = MDExFever.from_path(
    "datas/ex-fever/dev.csv",
    "datas/ex-fever/wiki_db.db",
)
output_path = "result/decompose/exfever.csv"

NUM_WORKERS = 4


async def worker(semaphore, decompose_wf, sample):
    async with semaphore:
        sample_dict = sample.model_dump()
        try:
            result = await decompose_wf.run(claim=sample.claim)
            sample_dict["context"] = "\n\n".join(sample_dict["context"])
            sample_dict["sub_claims"] = result.sub_claims
        except Exception as e:
            print(f"BUG: {e}")
            sample_dict["sub_claims"] = "BUG"
        return sample_dict


async def async_main():
    decompose_wf = DecomposeWorkflow(llm=llm, timeout=3000)
    semaphore = asyncio.Semaphore(NUM_WORKERS)

    batch_size = 50
    start = 2550
    first_batch = True if start == 0 else False

    for batch_start in range(start, len(dataset), batch_size):
        batch = []
        load_bug_results = []
        for i in range(batch_start, min(batch_start + batch_size, len(dataset))):
            try:
                batch.append(dataset[i])
            except Exception as e:
                print(f"BUG loading index {i}: {e}")
                load_bug_results.append({"claim": f"index_{i}", "sub_claims": "BUG"})

        tasks = [worker(semaphore, decompose_wf, sample) for sample in batch]
        results = await atqdm.gather(*tasks, desc=f"Batch {batch_start // batch_size + 1}")
        results = list(results) + load_bug_results

        batch_df = pd.DataFrame(results)
        if first_batch:
            batch_df.to_csv(output_path, index=False)
            first_batch = False
        else:
            batch_df.to_csv(output_path, mode='a', header=False, index=False)
        print(f"Saved rows {batch_start} - {batch_start + len(results) - 1}")


if __name__ == '__main__':
    asyncio.run(async_main())
