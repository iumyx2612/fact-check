import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from .datasets.base import LABELS


def evaluate_file(
        input_file: str,
        labels: list[str] = LABELS
):
    df = pd.read_csv(input_file)

    y_true = df["label"].tolist()
    y_pred = df["pred"].tolist()

    cls_report = classification_report(y_true, y_pred, labels=labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print(cls_report)
    print(matrix)
