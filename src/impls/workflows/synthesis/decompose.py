import re

from workflows import Workflow, step
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage

from ...events.synthesis.decompose import DecomposeStartEvent, DecomposeStopEvent
from src.modules.prompts.synthesis.decompose import DECOMPOSE_CLAIM_SYSTEM, DECOMPOSE_CLAIM_USER


class DecomposeWorkflow(Workflow):
    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def decompose(self, ev: DecomposeStartEvent) -> DecomposeStopEvent:
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

        return DecomposeStopEvent(sub_claims=sub_claims)