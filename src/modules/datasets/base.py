from typing import Optional

LABELS = ["SUPPORT", "REFUTE", "NEI"]


class Dataset:
    def __init__(
            self,
            claims: list[str],
            contexts: Optional[list[str]] = None,
            evidences: Optional[list[str]] = None,
            labels: Optional[list[str]] = None,
            **kwargs
    ):
        self.contexts = contexts
        self.claims = claims
        self.evidences = evidences
        self.labels = labels

    def __getitem__(self, index: int) -> dict:
        return {
            "context": self.contexts[index],
            "claim": self.claims[index] if self.claims else None,
            "evidence": self.evidences[index] if self.evidences else None,
            "label": self.labels[index] if self.labels else None
        }

    def __len__(self) -> int:
        return len(self.claims)
