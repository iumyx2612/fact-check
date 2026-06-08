import re
import random
from typing import Optional, Literal

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.prompts import ChatMessage

from src.modules.schema.graph_check.graph import Graph
from ...events.graph_check.context import VerifyContext
from ...events.graph_check.verify_graph import (
    VerifyGraphStartEvent,
    VerifyClaimLoopEvent,
    RetrieveEvidence,
    VerifyClaim,
    VerifyGraphStopEvent
)


def truncate(
        text: str,
        max_len: int = 40000
) -> str:
    if len(text) > max_len:
        text = text[:max_len]

    return text


def parse_boolean_answer(
        answer: str
) -> bool:

    answer = answer.split("\n")[0].lower().strip(" .")
    boolean_mapping = {
        "true": True, "false": False, "yes": True, "no": False,
        "it is impossible to say": False, "it's impossible to say": False,
        "it is impossible to tell": False, "it's impossible to tell": False,
        "it is not possible to say": False, "it's not possible to say": False,
        "it is not possible to tell": False, "it's not possible to tell": False
    }

    if answer in boolean_mapping:
        return boolean_mapping[answer]

    for sample_text, boolean_value in boolean_mapping.items():
        if answer.startswith(sample_text):
            return boolean_value

    return random.choice([True, False])



class VerifyGraphWorkflow(Workflow):
    def __init__(
            self,
            llm: LLM,
            graph: Graph,
            retriever: BaseRetriever,
            document_level: Literal[
                "concat",
                "each",
                "concat+each"
            ] = "concat+each",
            **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.graph = graph
        self.retriever = retriever
        self.document_level = document_level

    @step
    async def initialize(self, ctx: Context[VerifyContext], ev: VerifyGraphStartEvent) -> VerifyClaimLoopEvent:
        graph = self.graph

        total_triplets = graph.total_triples
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.all_triplets = total_triplets

        return VerifyClaimLoopEvent()

    @step
    async def loop_init(
            self, ctx: Context[VerifyContext], ev: VerifyClaimLoopEvent
    ) -> RetrieveEvidence | VerifyGraphStopEvent:
        all_triplets = await ctx.store.get("all_triplets")
        index = await ctx.store.get("index")
        prediction = await ctx.store.get("prediction")
        num_loops = len(all_triplets)

        if prediction == "REFUTE":
            return VerifyGraphStopEvent(prediction=prediction)

        if index >= num_loops:
            return VerifyGraphStopEvent(prediction=prediction)

        return RetrieveEvidence(claim=all_triplets[index])

    @step
    async def retrieve(
            self, ctx: Context[VerifyContext], ev: RetrieveEvidence
    ) -> VerifyClaim:
        nodes = self.retriever.retrieve(ev.claim)

        evidences = [node.text for node in nodes]
        return VerifyClaim(claim=ev.claim, evidences=evidences)

    @step
    async def verify(
            self, ctx: Context[VerifyContext], ev: VerifyClaim
    ) -> VerifyClaimLoopEvent:
        evidences = ev.evidences
        claim = ev.claim

        if self.document_level == "concat":
            evidence_str = truncate("\n".join(evidences))
            prompt = ChatMessage(
                content="Evidence: {evidence}\nClaim: {claim}\nIs the claim true or false?\nAnswer:".format(
                    evidence=evidence_str,
                    claim=claim
                ),
                role="user"
            )
            response = await self.llm.achat([prompt])
            answer = response.message.content
            answer = parse_boolean_answer(answer)

        elif self.document_level == "each":
            for i, evidence in enumerate(evidences):
                evidence_str = truncate(evidence)
                prompt = ChatMessage(
                    content="Evidence: {evidence}\nClaim: {claim}\nIs the claim true or false?\nAnswer:".format(
                        evidence=evidence_str,
                        claim=claim
                    ),
                    role="user"
                )
                response = await self.llm.achat([prompt])
                answer = response.message.content
                answer = parse_boolean_answer(answer)
                if answer:
                    break

        elif self.document_level == "concat+each":
            evidence_str = truncate("\n".join(evidences))
            prompt = ChatMessage(
                content="Evidence: {evidence}\nClaim: {claim}\nIs the claim true or false?\nAnswer:".format(
                    evidence=evidence_str,
                    claim=claim
                ),
                role="user"
            )
            response = await self.llm.achat([prompt])
            answer = response.message.content
            answer = parse_boolean_answer(answer)

            if not answer:
                for i, evidence in enumerate(evidences):
                    evidence_str = truncate(evidence)
                    prompt = ChatMessage(
                        content="Evidence: {evidence}\nClaim: {claim}\nIs the claim true or false?\nAnswer:".format(
                            evidence=evidence_str,
                            claim=claim
                        ),
                        role="user"
                    )
                    response = await self.llm.achat([prompt])
                    answer = response.message.content
                    answer = parse_boolean_answer(answer)
                    if answer:
                        break

        prediction = "SUPPORT" if answer else "REFUTE"

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.index += 1
            ctx_state.prediction = prediction

        return VerifyClaimLoopEvent()