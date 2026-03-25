from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator

from src.modules.graph_check.graph import Graph


class SynthesisContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    claim: str = ""
    path: list[str] = Field(default_factory=list)
    graph: Optional[Graph] = None
    infilled_def_triplets_texts: Optional[list[str]] = Field(default_factory=list)
    infilled_triplets_texts: Optional[list[str]] = Field(default_factory=list)
    infilling_log: list[dict] = Field(default_factory=list)
    current_latent_entity: Optional[str] = None
    infilling_index: int = Field(default=0, ge=0)
    path_index_for_trace: int = Field(default=0, ge=0)
    verification_index: int = Field(default=0, ge=0)
    verification_results: list[dict] = Field(default_factory=list)
    prediction: str = "SUPPORTED"

    @field_validator("infilling_index", "verification_index")
    @classmethod
    def validate_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Index cannot be negative")
        return v
