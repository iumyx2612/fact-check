from typing import Literal
from pydantic import BaseModel, Field

from src.modules.schema.graph_check.graph import Graph, Triplet


class SynthesisContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    # Const
    claim: str
    path: list[str]

    graph: Graph
    # Infilling
    infilled_def_triplets_texts: list[str] = Field(default=None)
    infilled_triplets_texts: list[str] = Field(default=None)
    infilling_log: list[dict] = Field(default=[])
    current_latent_entity: str = Field(default=None)
    infilling_index: int = Field(default=0)


class VerifyContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    all_triplets: list[Triplet]
    prediction: Literal["SUPPORT", "REFUTE", "NEI"] = "SUPPORT"
    index: int = 0
