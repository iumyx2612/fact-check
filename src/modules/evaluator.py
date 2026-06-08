import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from .datasets.base import LABELS


def levenshtein_distance(list1, list2):
    """
    Compute the Levenshtein distance between two sequences (lists) based on element-level edits.

    Args:
        list1 (list): The first sequence of elements.
        list2 (list): The second sequence of elements.

    Returns:
        int: The edit distance between the two sequences.
    """
    m = len(list1)
    n = len(list2)

    # Initialize DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming to/from empty sequence
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j],  # deletion
                    dp[i - 1][j - 1]  # substitution
                )

    return dp[m][n]


def char_sim(text1: str, text2: str) -> float:
    chars1 = list(text1.replace(" ", ""))
    chars2 = list(text2.replace(" ", ""))

    dist = levenshtein_distance(chars1, chars2)
    max_len = max(len(chars1), len(chars2))

    return 1 - dist / max_len


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


def evaluate(
        preds: list[str],
        gts: list[str],
        label = LABELS
):
    cls_report = classification_report(gts, preds, labels=label)
    matrix = confusion_matrix(gts, preds, labels=label)

    print(cls_report)
    print(matrix)