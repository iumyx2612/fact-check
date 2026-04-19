from typing import Literal
from pydantic import BaseModel

from llama_index.core.graph_stores.types import Triplet


class DecomposeContext(BaseModel):
    first_round_entities: list[str] = None
    first_round_triplets: list[Triplet] = None
    final_round_entities: list[str] = None
    final_round_triplets: list[Triplet] = None


class SynthesisContext(BaseModel):
    claim: str = None
    decompose_ctx: DecomposeContext = DecomposeContext()
    sub_claims: list[str] = None
    documents: list[str] = None
    sub_claim_index: int = 0
    sub_claim_mapping: dict[str, Literal["SUPPORT", "REFUTE", "NEI"]] = {}
