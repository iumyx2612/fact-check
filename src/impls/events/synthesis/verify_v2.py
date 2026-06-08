from typing import Literal
from workflows.events import StartEvent, Event, StopEvent
from llama_index.core.graph_stores.types import Triplet


class VerifyStartEvent(StartEvent):
    claim: str
    sub_claims: list[Triplet] | list[str]
    documents: list[str]


class BuildDocumentContext(Event):
    sub_claims: list[Triplet] | list[str]
    documents: list[str]


class VerifyLoopInit(Event):
    ...


class TryVerifyEvent(Event):
    sub_claim: str


class UnknownEntityResolutionLoop(Event):
    ...


class ExtractPossibleDocumentTriplets(Event):
    sub_claim: str


class ExtractRelevantDocumentSentences(Event):
    triplets: list[str]

class ExtractTripletsForUnknownEntity(Event):
    triplets_sentences_mapping: dict


class InfillDocumentTriplets(Event):
    triplets: list[str]
    triplets_sentences_mapping: dict


class MergeTriplets(Event):
    triplets: list[str]


class RemapSubClaim(Event):
    sub_claim: str


class TryInfillFullUnknowns(Event):
    sub_claim: str


class VerifyAggregateEvent(Event):
    ...


class VerifyStopEvent(StopEvent):
    result: Literal["SUPPORT", "REFUTE", "NEI"] = "SUPPORT"