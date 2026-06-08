from dotenv import load_dotenv
import pandas as pd
import json
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from ast import literal_eval
import asyncio
load_dotenv()
from workflows import Context
from llama_index.llms.openai import OpenAI

from src.impls.workflows.synthesis.decompose import DecomposeWorkflow
from src.impls.workflows.synthesis.verify import VerifyWorkflow, VerifyStartEvent
from src.impls.workflows.synthesis.context import SynthesisContext
from src.modules.datasets.ex_fever import DocDB


llm = OpenAI(model='gpt-4.1-mini', temperature=0)
docdb = DocDB("datas/ex-fever/wiki_db.db")
df = pd.read_csv("result/retrieval/ex-fever.csv")
output_path = "result/verify/exfever-rev-simple.csv"

NUM_WORKERS = 4


async def run_decompose(decompose_wf, claim):
    result = await decompose_wf.run(claim=claim)
    return result.sub_claims


async def run(wf, claim, documents, sub_claims, ctx):
    result = await wf.run(start_event=VerifyStartEvent(
        claim=claim,
        documents=documents,
        sub_claims=sub_claims,
    ), ctx=ctx)

    return result.result


async def worker(semaphore, decompose_wf, wf, sample):
    async with semaphore:
        try:
            claim = sample["claim"]
            # sub_claims = await run_decompose(decompose_wf, claim)
            sub_claims = literal_eval(sample["sub_claims"])
            doc_ids = literal_eval(sample["pred_doc_ids"])
            doc_ids = list(set(doc_ids))

            documents = []
            for doc_id in doc_ids:
                try:
                    documents.append(docdb.get_doc_text(doc_id))
                except Exception as e:
                    continue

            ctx = Context[SynthesisContext](wf)
            result = await run(wf, claim, documents, sub_claims, ctx)
            ctx_dict = ctx.to_dict()
            state = ctx_dict["state"]["state_data"]
            sub_claim_mapping = json.loads(state)["value"]["verify_ctx"]["verify_mapping"]
            docs = json.loads(state)["value"]["documents"]
            docs = "\n\n".join(docs)

            sample_dict = sample.to_dict()
            sample_dict["pred"] = result
            sample_dict["mapping"] = sub_claim_mapping
            sample_dict["pred_documents"] = docs
        except Exception as e:
            print(f"BUG: {e}")
            sample_dict = sample.to_dict()
            sample_dict["pred"] = "BUG"
            sample_dict["mapping"] = "BUG"
            sample_dict["pred_documents"] = "BUG"
        return sample_dict


async def async_main():
    decompose_wf = DecomposeWorkflow(llm=llm, timeout=3000, disable_validation=True)
    wf = VerifyWorkflow(llm=llm, use_reasoning=True, timeout=3000)
    semaphore = asyncio.Semaphore(NUM_WORKERS)

    batch_size = 50
    indices = list(range(len(df)))
    first_batch = True

    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        tasks = [worker(semaphore, decompose_wf, wf, df.iloc[i]) for i in batch_indices]
        results = await atqdm.gather(*tasks, desc=f"Batch {batch_start // batch_size + 1}")

        batch_df = pd.DataFrame(results)
        if first_batch:
            batch_df.to_csv(output_path, index=False)
            first_batch = False
        else:
            batch_df.to_csv(output_path, mode='a', header=False, index=False)
        print(f"Saved rows {batch_start} - {batch_start + len(results) - 1}")


def sync_main():
    wf = VerifyWorkflow(llm=llm, use_reasoning=True, timeout=3000, disable_validation=True)
    results = []

    for i in tqdm(range(len(df))):
        sample = df.iloc[i]

        uid = sample["uid"]
        # if uid != "265fe000-0bb9-4499-bd6f-282e9d7e4de5":
        # if uid != "748d3c5f-5e81-4d77-9ea6-82c5c53cac22":
        # if uid != "f64a887e-d4b1-452f-8b27-ea33e42614bb":
        if uid != "024d3fb0-2876-447c-ad66-cbcc61baeedc":
            continue

        claim = sample["claim"]
        # sub_claims = asyncio.run(run_decompose(decompose_wf, claim))
        sub_claims = literal_eval(sample["sub_claims"])
        documents = literal_eval(sample["pred_doc_ids"])
        documents = list(set(documents))
        documents = [docdb.get_doc_text(doc_id) for doc_id in documents]

        ctx = Context[SynthesisContext](wf)
        result = asyncio.run(run(wf, claim, documents, sub_claims, ctx))
        ctx_dict = ctx.to_dict()
        state = ctx_dict["state"]["state_data"]
        sub_claim_mapping = json.loads(state)["value"]["verify_ctx"]["verify_mapping"]
        docs = json.loads(state)["value"]["documents"]
        docs = "\n\n".join(docs)

        sample_dict = sample.to_dict()
        sample_dict["pred"] = result
        sample_dict["mapping"] = sub_claim_mapping
        sample_dict["pred_documents"] = docs
        results.append(sample_dict)

    new_df = pd.DataFrame(results)
    new_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    # sync_main()
    asyncio.run(async_main())
