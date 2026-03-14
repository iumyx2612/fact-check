"""
Comprehensive evaluation for GraphCheck benchmark results.

Provides detailed metrics including accuracy, precision, recall, F1 scores,
and per-class breakdown using sklearn.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)


def evaluate_graphcheck_results(csv_path: str):
    """
    Evaluate GraphCheck benchmark results with comprehensive metrics.

    Args:
        csv_path: Path to results CSV file with columns: claim, explanation, label, pred, is_correct
    """
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("GRAPHCHECK BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"\nDataset: {csv_path}")
    print(f"Total samples: {len(df)}")

    # Basic statistics
    print("\n" + "-" * 80)
    print("BASIC STATISTICS")
    print("-" * 80)
    print(f"Correct predictions: {df['is_correct'].sum()}/{len(df)}")
    print(f"Accuracy: {df['is_correct'].mean():.4f}")

    # Label distribution
    print("\n" + "-" * 80)
    print("LABEL DISTRIBUTION")
    print("-" * 80)
    print("\nGround Truth:")
    print(df['label'].value_counts().to_string())
    print("\nPredictions:")
    print(df['pred'].value_counts().to_string())

    # Classification report
    print("\n" + "-" * 80)
    print("CLASSIFICATION REPORT")
    print("-" * 80)

    labels = ['SUPPORT', 'REFUTE', 'NEI']
    y_true = df['label'].values
    y_pred = df['pred'].values

    # Get full classification report
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        zero_division=0,
        digits=4
    )
    print(report)

    # Detailed metrics
    print("\n" + "-" * 80)
    print("DETAILED METRICS")
    print("-" * 80)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        zero_division=0
    )

    print(f"\n{'Label':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, label in enumerate(labels):
        print(f"{label:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    print("\n" + "-" * 60)
    print(f"{'Macro Avg':<12} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {np.sum(support):<10}")
    print(f"{'Weighted Avg':<12} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {np.sum(support):<10}")
    print(f"{'Accuracy':<12} {accuracy:<12.4f} {'':<12} {'':<12} {np.sum(support):<10}")

    # Confusion matrix
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nPredicted →")
    print("Actual ↓")
    print(f"{'':>12}", end="")
    for label in labels:
        print(f"{label:>12}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"{label:>12}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>12}", end="")
        print()

    # Per-class analysis
    print("\n" + "-" * 80)
    print("PER-CLASS ANALYSIS")
    print("-" * 80)
    for label in labels:
        mask = df['label'] == label
        if mask.sum() > 0:
            correct = (df[mask]['pred'] == label).sum()
            total = mask.sum()
            print(f"\n{label}:")
            print(f"  Total samples: {total}")
            print(f"  Correct: {correct}")
            print(f"  Incorrect: {total - correct}")
            print(f"  Accuracy: {correct/total:.4f}")

            # Show misclassified samples
            misclassified = df[mask & (df['pred'] != label)]
            if len(misclassified) > 0:
                print(f"  Misclassified as:")
                for pred_label in misclassified['pred'].unique():
                    count = (misclassified['pred'] == pred_label).sum()
                    print(f"    {pred_label}: {count}")

    # Error analysis
    print("\n" + "-" * 80)
    print("ERROR ANALYSIS")
    print("-" * 80)
    errors = df[~df['is_correct']]
    if len(errors) > 0:
        print(f"\nTotal errors: {len(errors)}")
        print("\nError breakdown:")
        for true_label in labels:
            for pred_label in labels:
                if true_label != pred_label:
                    count = ((df['label'] == true_label) & (df['pred'] == pred_label)).sum()
                    if count > 0:
                        print(f"  {true_label} → {pred_label}: {count}")

        print("\nSample errors:")
        for idx, row in errors.head(3).iterrows():
            print(f"\n  Error {idx + 1}:")
            print(f"    Claim: {row['claim'][:100]}...")
            print(f"    True: {row['label']}, Pred: {row['pred']}")
            print(f"    Explanation: {row['explanation'][:150]}...")
    else:
        print("\nNo errors found!")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/home/an/code/fact-check/result/exfever-graphcheck-detailed.csv"

    evaluate_graphcheck_results(csv_path)