import re

from workflows import Workflow, step
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage

from ...events.synthesis.decompose import (
    DecomposeStartEvent,
    DecomposeVerifyEvent,
    DecomposeStopEvent
)
from src.modules.schema.synthesis.decompose import SubClaimVerdict
from src.modules.prompts.synthesis.decompose import (
    DECOMPOSE_CLAIM_SYSTEM,
    DECOMPOSE_CLAIM_USER,
    DECOMPOSE_VERIFY_SYSTEM,
    DECOMPOSE_VERIFY_USER
)


class DecomposeWorkflow(Workflow):
    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.verify_llm = llm.as_structured_llm(SubClaimVerdict)

    @step
    async def decompose(self, ev: DecomposeStartEvent) -> DecomposeVerifyEvent:
        claim = ev.claim

        response = await self.llm.achat([
            ChatMessage(
                content=DECOMPOSE_CLAIM_SYSTEM,
                role="system"
            ),
            ChatMessage(
                content=DECOMPOSE_CLAIM_USER.format(claim=claim),
                role="user"
            )
        ])

        content = response.message.content
        sub_claims = []
        for line in content.strip().splitlines():
            line = line.strip()
            match = re.match(r"^\d+\.\s+(.+)$", line)
            if match:
                sub_claims.append(match.group(1))

        return DecomposeVerifyEvent(
            sub_claims=sub_claims,
            claim=claim
        )

    @step
    async def verify(self, ev: DecomposeVerifyEvent) -> DecomposeStopEvent:
        claim = ev.claim
        sub_claims = ev.sub_claims
        new_subclaims = []

        for sub_claim in sub_claims:
            response = await self.verify_llm.achat([
                ChatMessage(
                    content=DECOMPOSE_VERIFY_SYSTEM,
                    role="system"
                ),
                ChatMessage(
                    content=DECOMPOSE_VERIFY_USER.format(
                        claim=claim,
                        sub_claim=sub_claim
                    ),
                    role="user"
                )
            ])
            verdict = response.raw

            if verdict.correct:
                new_subclaims.append(sub_claim)
            else:
                if not verdict.correct_sub_claim:
                    continue
                new_subclaims.append(verdict.correct_sub_claim)

        return DecomposeStopEvent(sub_claims=new_subclaims)
