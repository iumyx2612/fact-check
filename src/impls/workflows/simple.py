import re

from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatMessage
from workflows import Workflow, step
from workflows.events import StopEvent

from src.modules.prompts.simple import SIMPLE_USER, SIMPLE_REASONING_USER
from ..events.base import FactCheckStartEvent


class SimpleBaseFactCheck(Workflow):
    def __init__(
            self,
            llm: LLM,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def fact_check(self, ev: FactCheckStartEvent) -> StopEvent:
        context = ev.context
        claim = ev.claim
        prompt = ChatMessage(
            content=SIMPLE_USER.format(
                context=context,
                claim=claim
            ),
            role="user"
        )

        response = await self.llm.achat([prompt])
        label = response.message.content

        # Convert label
        if label.lower() == "yes":
            label = "SUPPORT"
        elif label.lower() == "no":
            label = "REFUTE"
        elif label.lower() == "not enough information":
            label = "NEI"

        return StopEvent(label)


class SimpleReasoningFactCheck(Workflow):
    def __init__(
            self,
            llm: LLM,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def fact_check(self, ev: FactCheckStartEvent) -> StopEvent:
        context = ev.context
        claim = ev.claim
        prompt = ChatMessage(
            content=SIMPLE_REASONING_USER.format(
                context=context,
                claim=claim
            ),
            role="user"
        )

        response = await self.llm.achat([prompt])
        content = response.message.content

        # Parse
        pattern = re.compile(
            r"Reasoning:\s*(?P<reasoning>.*?)\s*Answer:\s*(?P<answer>Not Enough Information|Yes|No)",
            re.DOTALL
        )

        match = pattern.search(content)

        label = "Bug"
        reasoning = "Bug"
        if match:
            reasoning = match.group("reasoning").strip()
            answer = match.group("answer")

            # Convert label
            if answer.lower().strip() == "yes":
                label = "SUPPORT"
            elif answer.lower().strip() == "no":
                label = "REFUTE"
            elif answer.lower().strip() == "not enough information":
                label = "NEI"

        return StopEvent({
            "label": label,
            "reasoning": reasoning
        })