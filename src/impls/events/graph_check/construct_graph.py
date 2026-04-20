from workflows.events import StartEvent, StopEvent, Event

from src.modules.graph_check.graph import Graph


class ConstructGraphStartEvent(StartEvent):
    claim: str


class ParseGraphEvent(Event):
    content: str


class ConstructGraphStopEvent(StopEvent):
    graph: Graph