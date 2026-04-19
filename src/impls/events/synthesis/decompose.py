from workflows.events import StartEvent, StopEvent


class DecomposeStartEvent(StartEvent):
    claim: str


class DecomposeStopEvent(StopEvent):
    sub_claims: list[str]