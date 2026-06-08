from workflows.events import StartEvent, StopEvent, Event
from llama_index.core.graph_stores.types import Triplet


class DecomposeStartEvent(StartEvent):
    claim: str


class DecomposeExtractKnownEntities(Event):
    entities: list[str]


class DecomposeExtractRelation(Event):
    entities: list[str]


class DecomposeGleaningEntity(Event):
    existing_triplets: list[Triplet]


class DecomposeMergeGraph(Event):
    all_triplets: list[Triplet]


class DecomposeVerifyGraph(Event):
    all_triplets: list[Triplet]


class DecomposeStopEvent(StopEvent):
    sub_claims: list[Triplet]