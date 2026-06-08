from typing import Literal
from pydantic import BaseModel

from llama_index.core.graph_stores.types import Triplet


class DecomposeContext(BaseModel):
    first_round_entities: list[str] = None
    first_round_triplets: list[Triplet] = None
    final_round_entities: list[str] = None
    final_round_triplets: list[Triplet] = None


class DocumentContext(BaseModel):
    triplets: list[str] = []
    known_entities: list[str] = []
    unknown_entities: list[str] = []
    known_entity_index: int = 0


class VerifyContext(BaseModel):
    verify_index: int = 0
    current_sub_claim: str = ""
    known_entities: list[str] = []
    unknown_entities: list[str] = []
    remapped_triplets: list[str] = []
    remapped_sub_claims: list[str] = {}
    remained_subclaims: list[str] = []
    verify_counter: dict[str, int] = {}
    verify_mapping: dict[str, Literal["SUPPORT", "REFUTE", "NEI"]] = {}


class SynthesisContext(BaseModel):
    # Shared
    claim: str = None
    sub_claims: list[str] = None
    documents: list[str] = None

    decompose_ctx: DecomposeContext = DecomposeContext()
    document_ctx: DocumentContext = DocumentContext()
    verify_ctx: VerifyContext = VerifyContext()
