import pandas as pd

from .base import Dataset

LABEL_MAPPING = {
    "Supports": "SUPPORT",
    "Refutes": "REFUTE",
    "Not_Enough_Information": "NEI"
}


class ViWiKiFC(Dataset):

    @classmethod
    def from_csv(cls, path: str):
        df = pd.read_csv(path)

        contexts = df["context"].tolist()
        claims = df["claim"].tolist()
        evidences = df["evidence"].tolist()
        i_labels = df["gold_label"].tolist()
        labels = [LABEL_MAPPING[i] for i in i_labels]

        return cls(contexts, claims, evidences, labels)