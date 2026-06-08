from workflows.events import StartEvent, StopEvent, Event


class DecomposeStartEvent(StartEvent):
    claim: str


class DecomposeVerifyEvent(Event):
    claim: str
    sub_claims: list[str]


class DecomposeStopEvent(StopEvent):
    sub_claims: list[str]