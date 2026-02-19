from typing import Optional
from workflows.events import StartEvent, Event, StopEvent
from pydantic import BaseModel

from src.modules.schema.graph_check.graph import Graph


class InfillingStartEvent(StartEvent):
    claim: str
    path: list[str]


class InfillingLoopInitialize(Event):
    ...


class MakeInfillingRetrievalQuery(Event):
    ...


class RetrieveEvidenceEvent(Event):
    query: str


class MakeInfillingQuery(Event):
    ...


class InfillEvent(Event):
    infill_query: Optional[str] = None
    evidence: Optional[str] = None


class HandleLoopInfo(Event):
    infill: str
    query: str


class InfillingStopEvent(StopEvent):
    graph: Graph