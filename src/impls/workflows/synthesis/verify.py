import re

from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage

from .context import SynthesisContext
from ...events.synthesis.verify import (
    VerifyStartEvent,
    VerifyLoopInit,
    VerifySingleSubClaim,
    VerifyAggregateEvent,
    VerifyStopEvent
)
from src.modules.prompts.synthesis.verify import (
    VERIFY_SYSTEM,
    VERIFY_USER,
    VERIFY_REASONING_SYSTEM,
    VERIFY_REASONING_SYSTEM_V2,
    VERIFY_REASONING_USER
)
from src.modules.datasets.base import LABELS


class VerifyWorkflow(Workflow):
    def __init__(self,
                 llm: LLM,
                 use_reasoning: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_reasoning = use_reasoning
        self.llm = llm

    @step
    async def initialize(
            self, ctx: Context[SynthesisContext], ev: VerifyStartEvent
    ) -> VerifyLoopInit:
        sub_claims = ev.sub_claims

        if not isinstance(sub_claims[0], str):
            sub_claims = "\n".join([
                f"{sub_claim[0].id} -> {sub_claim[1].id} -> {sub_claim[2].id}"
            for sub_claim in sub_claims])

        documents = ev.documents

        async with ctx.store.edit_state() as ctx_state:
            ctx_state.sub_claims = sub_claims
            ctx_state.documents = documents

        return VerifyLoopInit()

    @step
    async def loop_init(
            self, ctx: Context[SynthesisContext], ev: VerifyLoopInit
    ) -> VerifySingleSubClaim | VerifyAggregateEvent:
        verify_index = await ctx.store.get("verify_ctx.verify_index")
        sub_claims = await ctx.store.get("sub_claims")

        if verify_index >= len(sub_claims):
            return VerifyAggregateEvent()

        return VerifySingleSubClaim(sub_claim=sub_claims[verify_index])

    @step
    async def verify(
            self, ctx: Context[SynthesisContext], ev: VerifySingleSubClaim
    ) -> VerifyLoopInit | VerifyAggregateEvent:
        sub_claim = ev.sub_claim
        documents = await ctx.store.get("documents")

        if self.use_reasoning:
            response = await self.llm.achat([
                ChatMessage(
                    content=VERIFY_REASONING_SYSTEM_V2,
                    role="system"
                ),
                ChatMessage(
                    content=VERIFY_REASONING_USER.format(
                        document="\n\n".join(documents),
                        claim=sub_claim
                    ),
                    role="user"
                )
            ])
            content = response.message.content
            # Regex to extract Verdict
            match = re.search(r"verdict:\s*(true|false|not enough information)", content, re.IGNORECASE)
            verdict_str = match.group(1).lower() if match else "not enough information"

            verdict_map = {
                "true": LABELS[0],
                "false": LABELS[1],
                "not enough information": LABELS[2],
            }
            label = verdict_map.get(verdict_str, LABELS[2])
        else:
            response = await self.llm.achat([
                ChatMessage(
                    content=VERIFY_SYSTEM,
                    role="system"
                ),
                ChatMessage(
                    content=VERIFY_USER.format(
                        document="\n\n".join(documents),
                        claim=sub_claim
                    ),
                    role="user"
                )
            ])
            content = response.message.content

            if "True" in content:
                label = "SUPPORT"
            elif "False" in content:
                label = "REFUTE"
            elif "Not Enough Information" in content:
                label = "NEI"
            else:
                label = "NEI"

        # Update context
        async with ctx.store.edit_state() as ctx_state:
            ctx_state.verify_ctx.verify_mapping[sub_claim] = label
            ctx_state.verify_ctx.verify_index += 1

        # Early stopping
        if label == "REFUTE":
            return VerifyAggregateEvent()

        return VerifyLoopInit()

    @step
    async def aggregate(self, ctx: Context[SynthesisContext], ev: VerifyAggregateEvent) -> VerifyStopEvent:
        sub_claim_mapping = await ctx.store.get("verify_ctx.verify_mapping")
        for sub_claim, result in sub_claim_mapping.items():
            if result == "REFUTE":
                return VerifyStopEvent(result="REFUTE")
            elif result == "NEI": # TODO: Need more investigation
                return VerifyStopEvent(result="NEI")

        return VerifyStopEvent(result="SUPPORT")