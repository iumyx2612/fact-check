from typing import Optional, Literal
from pydantic import BaseModel

LABELS = ["SUPPORT", "REFUTE", "NEI"]


class DatasetOutput(BaseModel):
    model_config = {"extra": "allow"}

    claim: str
    label: Optional[Literal["SUPPORT", "REFUTE", "NEI"]] = None
    context: Optional[list[str]] = None
    evidence: Optional[list[str]] = None
    golden_docs: Optional[list[str]] = None


class Dataset:
    def __init__(
            self,
            claims: list[str],
            contexts: Optional[list[str]] = None,
            evidences: Optional[list[str]] = None,
            labels: Optional[list[str]] = None,
            golden_docs: Optional[list[list[str]]] = None,
            **kwargs
    ):
        self.contexts = contexts
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.golden_docs = golden_docs

    def __getitem__(self, index: int) -> DatasetOutput:
        return DatasetOutput(
            claim=self.claims[index],
            label=self.labels[index] if self.labels else None,
            context=self.contexts[index] if self.contexts else None,
            evidence=self.evidences[index] if self.evidences else None,
            golden_docs=self.golden_docs[index] if self.golden_docs else None,
        )

    def __len__(self) -> int:
        return len(self.claims)
