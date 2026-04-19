from typing import Literal
from workflows.events import StartEvent, Event, StopEvent
from llama_index.core.graph_stores.types import Triplet


class VerifyStartEvent(StartEvent):
    sub_claims: list[str] | list[Triplet]
    claim: str
    documents: list[str]


class VerifyLoopInit(Event):
    ...


class VerifySingleSubClaim(Event):
    sub_claim: str


class VerifyAggregateEvent(Event):
    ...


class VerifyStopEvent(StopEvent):
    result: Literal["SUPPORT", "REFUTE", "NEI"]