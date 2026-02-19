from workflows import Workflow, step, Context
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage

from ..events.summary import SummaryStartEvent, StopEvent
from src.modules.prompts.summary import SUMMARY_SYSTEM, SUMMARY_EXAMPLE, SUMMARY_USER


class SummaryWorkflow(Workflow):
    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def summary(self, ev: SummaryStartEvent) -> StopEvent:
        system_prompt = ChatMessage(
            content=SUMMARY_SYSTEM.format(
                example=SUMMARY_EXAMPLE
            ),
            role="system"
        )
        user_prompt = ChatMessage(
            content=SUMMARY_USER.format(
                convo=ev.convo,
            ),
            role="user"
        )

        response = await self.llm.achat([system_prompt, user_prompt])

        content = response.message.content

        return StopEvent(content)