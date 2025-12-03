"""
Multivariate modelling and classification utilities.

This module contains functions to perform principal component analysis (PCA)
for dimensionality reduction and visualisation, linear discriminant analysis
(LDA) and logistic regression for classification.  It also provides tools
for cross‑validation across topics to prevent information leak between
training and test sets.

The command line interface allows users to run PCA and evaluate models
directly from the command line.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler

__all__ = [
    "run_pca",
    "evaluate_classifiers",
]


def run_pca(df: pd.DataFrame, label_col: str = "label", n_components: int = 2) -> Tuple[pd.DataFrame, PCA]:
    """Perform PCA on numeric features and return component scores.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numeric feature columns and a label column.
    label_col : str, default 'label'
        Name of the label column which will be excluded from PCA.
    n_components : int, default 2
        Number of principal components to compute.

    Returns
    -------
    (scores, pca) : Tuple[pandas.DataFrame, sklearn.decomposition.PCA]
        A DataFrame with component scores and the fitted PCA object.
    """
    # Identify numeric columns
    numeric_cols = [c for c in df.columns if c not in [label_col, "topic"] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(X_scaled)
    comp_df = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n_components)])
    comp_df[label_col] = df[label_col].values
    if "topic" in df.columns:
        comp_df["topic"] = df["topic"].values
    return comp_df, pca


def evaluate_classifiers(
    df: pd.DataFrame,
    label_col: str = "label",
    topic_col: Optional[str] = "topic",
    n_splits: int = 5,
) -> Tuple[dict, dict]:
    """Train LDA and logistic regression models with cross‑validation.

    Splits the data by topic if a topic column is provided, otherwise uses
    stratified k‑fold.  Returns dictionaries containing ROC and PR curve
    metrics for each fold and for each classifier.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with numeric features and labels.
    label_col : str, default 'label'
        Name of the label column.
    topic_col : str or None, default 'topic'
        If provided and present in df, use `GroupKFold` to avoid training and
        testing on the same topic.
    n_splits : int, default 5
        Number of folds for cross‑validation.

    Returns
    -------
    (roc_results, pr_results) : Tuple[dict, dict]
        Dictionaries keyed by model name containing lists of fold metrics
        (fpr, tpr, roc_auc) and (precision, recall, average_precision).
    """
    numeric_cols = [c for c in df.columns if c not in [label_col, topic_col] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].values
    y = df[label_col].astype(str).values
    # Binarise labels: assume two unique values
    labels = np.unique(y)
    if len(labels) != 2:
        raise ValueError("Binary classification expected for evaluate_classifiers")
    y_bin = (y == labels[1]).astype(int)  # treat second label as positive
    # Cross‑validation strategy
    if topic_col and topic_col in df.columns:
        groups = df[topic_col].values
        cv = GroupKFold(n_splits=n_splits)
        splits = cv.split(X, y_bin, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = cv.split(X, y_bin)
    # Models
    lda = LinearDiscriminantAnalysis()
    logreg = LogisticRegression(max_iter=1000)
    models = {"LDA": lda, "Logistic": logreg}
    roc_results = {k: [] for k in models}
    pr_results = {k: [] for k in models}
    scaler = StandardScaler()
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_bin[train_idx], y_bin[test_idx]
        # Standardise
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        for name, model in models.items():
            model.fit(X_train_std, y_train)
            prob = model.predict_proba(X_test_std)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            roc_results[name].append({"fpr": fpr, "tpr": tpr, "auc": roc_auc})
            pr_results[name].append({"precision": precision, "recall": recall, "ap": ap})
    return roc_results, pr_results


def plot_pca_scatter(comp_df: pd.DataFrame, label_col: str = "label", out_path: str = "pca_scatter.png") -> None:
    """Generate and save a scatter plot of the first two principal components."""
    if "PC1" not in comp_df.columns or "PC2" not in comp_df.columns:
        raise KeyError("PCA results must contain 'PC1' and 'PC2'")
    plt.figure(figsize=(6, 5))
    label_map = {'human': 'Humano', 'llm': 'LLM'}
    for label in comp_df[label_col].unique():
        subset = comp_df[comp_df[label_col] == label]
        display_label = label_map.get(str(label).lower(), str(label))
        plt.scatter(subset["PC1"], subset["PC2"], label=display_label, alpha=0.6, s=20)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Gráfico de Dispersão PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PCA or classify using LDA/logistic regression.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pca_parser = subparsers.add_parser("pca", help="Perform PCA on feature dataset.")
    pca_parser.add_argument("--features", required=True, help="Path to CSV with extracted features.")
    pca_parser.add_argument("--label-col", default="label", help="Name of the label column.")
    pca_parser.add_argument("--n-components", type=int, default=2, help="Number of principal components.")
    pca_parser.add_argument("--out", default="pca_scores.csv", help="Path to save PCA scores.")
    pca_parser.add_argument("--plot", default="pca_scatter.png", help="Path to save PCA scatter plot.")

    cls_parser = subparsers.add_parser("classify", help="Evaluate LDA and logistic regression models.")
    cls_parser.add_argument("--features", required=True, help="Path to CSV with extracted features.")
    cls_parser.add_argument("--label-col", default="label", help="Name of the label column.")
    cls_parser.add_argument("--topic-col", default="topic", help="Name of the topic column (optional).")
    cls_parser.add_argument("--n-splits", type=int, default=5, help="Number of cross‑validation folds.")
    cls_parser.add_argument("--roc-out", default="roc_results.pkl", help="Path to save ROC results (pickle).")
    cls_parser.add_argument("--pr-out", default="pr_results.pkl", help="Path to save PR results (pickle).")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "pca":
        df = pd.read_csv(args.features)
        scores, pca = run_pca(df, args.label_col, args.n_components)
        scores.to_csv(args.out, index=False)
        if args.n_components >= 2:
            plot_pca_scatter(scores, args.label_col, args.plot)
    elif args.command == "classify":
        df = pd.read_csv(args.features)
        roc_results, pr_results = evaluate_classifiers(df, args.label_col, args.topic_col, args.n_splits)
        # Persist results using pandas to pickle (not executed here)
        import pickle
        with open(args.roc_out, "wb") as f:
            pickle.dump(roc_results, f)
        with open(args.pr_out, "wb") as f:
            pickle.dump(pr_results, f)


if __name__ == "__main__":
    main()