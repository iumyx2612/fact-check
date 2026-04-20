from typing import Optional
from workflows.events import StartEvent, Event, StopEvent
from pydantic import BaseModel

from src.modules.graph_check.graph import Graph


class InfillingStartEvent(StartEvent):
    claim: str
    path: list[str]
    graph: Graph
    path_index: int = 0


class InfillingLoopInitialize(Event):
    ...


class MakeInfillingRetrievalQuery(Event):
    ...


class RetrieveEvidenceEvent(Event):
    query: str


class MakeInfillingQuery(Event):
    ...


class InfillQueryEvent(Event):
    """Event containing infilling query and retrieval query, before evidence retrieval."""
    infill_query: Optional[str] = None
    retrieval_query: Optional[str] = None


class InfillEvent(Event):
    """Event containing infilling query and retrieved evidence, ready for infilling."""
    infill_query: Optional[str] = None
    retrieval_query: Optional[str] = None
    evidence: Optional[str] = None


class HandleLoopInfo(Event):
    infill: str
    query: str


class InfillingStopEvent(StopEvent):
    graph: Graph
    infilling_log: list[dict]