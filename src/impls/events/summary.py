from workflows.events import StartEvent, Event, StopEvent


class SummaryStartEvent(StartEvent):
    convo: str
