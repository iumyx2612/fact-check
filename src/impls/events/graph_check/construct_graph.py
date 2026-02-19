from workflows.events import StartEvent, StopEvent, Event

from src.modules.schema.graph_check.graph import Graph


class ConstructGraphStartEvent(StartEvent):
    claim: str


class ParseGraphEvent(Event):
    content: str


class ConstructGraphStopEvent(StopEvent):
    graph: Graph