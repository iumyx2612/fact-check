from workflows.events import StartEvent, StopEvent, Event


class RetrievalStartEvent(StartEvent):
    claim: str
    subclaims: list[str]


class RetrieveEntityEvent(Event):
    claim: str


class RetrieveTripletsEvent(Event):
    subclaims: list[str]


class RetrievalAggregateEvent(Event):
    doc_ids: list[str]


class RetrievalStopEvent(StopEvent):
    doc_ids: list[str]