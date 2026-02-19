from pydantic import BaseModel, Field

from src.modules.schema.graph_check.graph import Graph


class SynthesisContext(BaseModel):
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
