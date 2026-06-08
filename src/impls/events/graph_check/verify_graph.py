from typing import Optional, Any, Literal
from workflows.events import StartEvent, Event, StopEvent


class VerifyGraphStartEvent(StartEvent):
    ...


class VerifyClaimLoopEvent(Event):
    ...


class RetrieveEvidence(Event):
    claim: str
    top_k: int


class VerifyClaim(Event):
    claim: str
    evidences: list[str]


class VerifyGraphStopEvent(StopEvent):
    prediction: Literal["SUPPORT", "REFUTE", "NEI"] = "SUPPORT"
