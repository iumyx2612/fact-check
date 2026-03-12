from typing import Optional, Any
from pydantic import BaseModel, Field

from src.modules.schema.graph_check.graph import Graph


class SynthesisContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    claim: str = ""
    path: list[str] = Field(default_factory=list)
    graph: Optional[Graph] = None
    infilled_def_triplets_texts: Optional[list[str]] = Field(default_factory=list)
    infilled_triplets_texts: Optional[list[str]] = Field(default_factory=list)
    infilling_log: list[dict] = Field(default_factory=list)
    current_latent_entity: Optional[str] = None
    infilling_index: int = 0
    verification_index: int = 0
    verification_results: list[dict] = Field(default_factory=list)
    prediction: str = "SUPPORTED"
