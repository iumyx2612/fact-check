from workflows.events import StartEvent, StopEvent


class FactCheckStartEvent(StartEvent):
    context: str
    claim: str