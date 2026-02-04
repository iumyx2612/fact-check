import pandas as pd

from .base import Dataset

LABEL_MAPPING = {
    0: "SUPPORT",
    1: "REFUTE",
    2: "NEI"
}


class ViFactCheck(Dataset):

    @classmethod
    def from_csv(cls, path: str):
        df = pd.read_csv(path)

        contexts = df["Context"].tolist()
        claims = df["Statement"].tolist()
        evidences = df["Evidence"].tolist()
        i_labels = df["labels"].tolist()
        labels = [LABEL_MAPPING[i] for i in i_labels]

        return cls(contexts, claims, evidences, labels)