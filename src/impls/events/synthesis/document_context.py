from workflows.events import StartEvent, Event, StopEvent
from llama_index.core.graph_stores.types import Triplet


class DocumentContextStartEvent(StartEvent):
    sub_claims: list[Triplet] | list[str]
    documents: list[str]


class ExtractKnownEntitiesFromClaim(Event):
    sub_claims: list[str]


class KnownEntityGraphInit(Event):
    ...


class ExtractSentencesFromKnownEntity(Event):
    entity: str


class ReplaceEntityIntoSentences(Event):
    sentences: list[str]
    entity: str


class ExtractTripletFromSentences(Event):
    sentences: list[str]
    entity: str


class MergeKnownTriplets(Event):
    triplets: list[str]


class RemoveDuplicatedKnownTriplets(Event):
    ...


class DocumentContextStopEvent(StopEvent):
    triplets: list[str]
    known_entities: list[str] = []
    unknown_entities: list[str] = []
