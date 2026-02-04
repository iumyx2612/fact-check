from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import asyncio

from llama_index.llms.openai import OpenAI

from src.impls.events.base import FactCheckStartEvent
from src.impls.workflows.simple import SimpleBaseFactCheck, SimpleReasoningFactCheck
from src.modules.datasets.vifactcheck import ViFactCheck
from src.modules.datasets.viwikifc import ViWiKiFC
from src.modules.evaluator import evaluate_file

load_dotenv()


dataset = ViFactCheck.from_csv("datas/vifactcheck/test.csv")
llm = OpenAI(model="gpt-4.1-mini")

# ==========================================
wf = SimpleBaseFactCheck(llm=llm)
# ==========================================


async def benchmark(output_file: str):
    out_df = pd.DataFrame()
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        context = sample["context"]
        claim = sample["claim"]
        evidence = sample["evidence"]
        label = sample["label"]

        start_ev = FactCheckStartEvent(context=context, claim=claim)
        output = await wf.run(start_event=start_ev)
        prediction = str(output)
        # prediction = output["label"]
        # reasoning = output["reasoning"]

        out_df = out_df._append({
            "context": context,
            "claim": claim,
            "evidence": evidence,
            "label": label,
            "pred": prediction,
            # "reasoning": reasoning
        }, ignore_index=True)

    out_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    asyncio.run(benchmark("result/vifactcheck-simple(2).csv"))
    evaluate_file("result/vifactcheck-simple(2).csv")