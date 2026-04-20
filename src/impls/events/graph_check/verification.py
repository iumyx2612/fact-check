from typing import Optional
from workflows.events import StartEvent, Event, StopEvent
from pydantic import BaseModel

from src.modules.graph_check.graph import Graph


class VerificationStartEvent(StartEvent):
    graph: Graph
    path_index: int = 0


class VerificationLoopInitialize(Event):
    ...


class VerifyTripleEvent(Event):
    triple_text: str


class VerificationStopEvent(StopEvent):
    prediction: str
    verification_results: list[dict]
