 
from __future__ import annotations
 
import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_DATA_PATH = "data/cleaned_disease_symptom.csv"
TARGET_COLUMN = "diseases"
EXPECTED_N_FEATURES = 377  # expected specification - verified at runtime
EXPECTED_N_CLASSES = 773  # expected specification - verified at runtime
 
MODEL_FILENAME = "sanjeevani_model.pkl"
METADATA_FILENAME = "sanjeevani_model_metadata.json"
SCHEMA_FILENAME = "symptom_schema.json"
MODEL_VERSION = "0.1.0-prototype"
TOP_K_VALUES = (1, 3, 5)
DROPOUT_KEEP_RANGE = (1, 4)  # dropout copies keep this many symptoms per row (inclusive)
FEW_SYMPTOM_COUNTS = (4, 3, 2, 1)  # users describe fewer symptoms than dataset rows contain
SELECTION_FEW_SYMPTOMS = 3

DISCLAIMER = (
    "Development prototype. Output is a ranked list of candidate condition "
    "categories for intake support only - NOT a diagnosis, NOT a treatment or "
    "medication recommendation, and NOT a measure of clinical urgency. Scores are "
    "uncalibrated. A human clinician must review all cases; emergency recognition "
    "must be handled by a separate, clinically validated mechanism."
)
 
# Candidate models + small, editable hyperparameter grids.
# Selection uses a validation split carved out of the TRAINING data only.
MODEL_REGISTRY = {
    # The project's explainable classifier: every prediction traces to feature splits.
    # Sparse binary symptoms need deep trees, so depth is not capped. Leaves of 1 sample
    # are excluded because they report a false 100% score from a single training row.
    "decision_tree": (
        lambda rs, nj: DecisionTreeClassifier(criterion="entropy", random_state=rs),
        {"min_samples_leaf": [2, 3, 5]},
    ),
    # Binary symptom features -> Bernoulli NB models "symptom present / absent" directly.
    "bernoulli_nb": (
        lambda rs, nj: BernoulliNB(binarize=None),  # data is validated as 0/1 already
        {"alpha": [0.01, 0.1, 1.0], "fit_prior": [True, False]},
    ),
    "multinomial_nb": (
        lambda rs, nj: MultinomialNB(),
        {"alpha": [0.01, 0.1, 1.0], "fit_prior": [True, False]},
    ),
    # One-vs-rest linear SVM: 773 binary problems per fit -> can be slow.
    # dual=False (primal) is the right choice when n_samples >> n_features.
    "linear_svc": (
        lambda rs, nj: LinearSVC(dual=False, max_iter=2000, random_state=rs),
        {"C": [0.1, 1.0], "class_weight": [None, "balanced"]},
    ),
    # Multinomial LR with 773 classes is slow & memory-hungry. Opt-in only.
    "logistic_regression": (
        lambda rs, nj: LogisticRegression(solver="lbfgs", max_iter=300, random_state=rs),
        {"C": [1.0, 10.0]},
    ),
    # Trees store a value vector of length n_classes per node -> memory can explode
    # with 773 classes. Opt-in only; keep n_estimators / depth modest.
    "random_forest": (
        lambda rs, nj: RandomForestClassifier(
            n_estimators=100, max_features="sqrt", n_jobs=nj, random_state=rs
        ),
        {"min_samples_leaf": [2, 5]},
    ),
}
DEFAULT_MODELS = "decision_tree"
 
 
class DataValidationError(Exception):
    """Raised when the dataset has a problem that must be fixed before training."""
 
 
def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
 
 
# --------------------------------------------------------------------------- #
# STEP 1 - Loading
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Run from the project root or pass --data."
        )
    try:
        df = pd.read_csv(data_path, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not parse '{data_path}' as CSV: {exc}") from exc
    print(f"Loaded '{data_path}' -> shape {df.shape}")
    return df
 
 
def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
 
 
# --------------------------------------------------------------------------- #
# STEP 2/3 - Validation and feature-target separation
# --------------------------------------------------------------------------- #
def validate_data(df: pd.DataFrame, allow_spec_mismatch: bool = False):
    """Validate the dataset; return (X, y, report). Never drops rows or edits labels."""
    critical: list[str] = []
    warns: list[str] = []
    report: dict = {"n_rows": int(len(df)), "n_columns": int(df.shape[1])}
 
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(
            f"Target column '{TARGET_COLUMN}' not found. First columns: {list(df.columns[:10])}"
        )
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        critical.append(f"Duplicate column names found: {dup_cols[:10]}")
    if len(df) == 0:
        raise DataValidationError("Dataset has zero rows.")
 
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    print(f"Shape: {df.shape} | symptom feature columns: {len(feature_cols)}")
    print("Column dtypes:", df.dtypes.astype(str).value_counts().to_dict())
    print("First 5 columns:", list(df.columns[:5]))
 
    # --- missing values -------------------------------------------------------
    feature_nan = int(df[feature_cols].isna().sum().sum())
    target_nan = int(df[TARGET_COLUMN].isna().sum())
    blank_labels = int(df[TARGET_COLUMN].dropna().astype(str).str.strip().eq("").sum())
    report.update(feature_missing_cells=feature_nan, target_missing=target_nan, blank_labels=blank_labels)
    if feature_nan:
        critical.append(f"{feature_nan} missing values in symptom features.")
    if target_nan or blank_labels:
        critical.append(f"Target has {target_nan} missing and {blank_labels} blank labels.")
 
    # --- numeric / binary checks ---------------------------------------------
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        critical.append(f"{len(non_numeric)} non-numeric feature column(s), e.g. {non_numeric[:10]}")
    numeric_cols = [c for c in feature_cols if c not in non_numeric]
 
    non_binary: list[str] = []
    constant_cols: list[str] = []
    zero_symptom_rows = 0
    if numeric_cols:
        values = df[numeric_cols]
        bad = ~values.isin([0, 1]) & values.notna()
        non_binary = bad.columns[bad.any()].tolist()
        constant_cols = [c for c in numeric_cols if values[c].nunique(dropna=False) <= 1]
        zero_symptom_rows = int((values.fillna(0).sum(axis=1) == 0).sum())
    if non_binary:
        critical.append(
            f"{len(non_binary)} feature column(s) contain values other than 0/1, e.g. {non_binary[:10]}"
        )
    if constant_cols:
        warns.append(
            f"{len(constant_cols)} constant symptom column(s) (kept to preserve schema): {constant_cols[:10]}"
        )
    if zero_symptom_rows:
        warns.append(f"{zero_symptom_rows} row(s) have no symptom flagged at all.")
    report.update(
        non_numeric_columns=non_numeric,
        non_binary_columns=non_binary,
        constant_columns=constant_cols,
        zero_symptom_rows=zero_symptom_rows,
    )
 
    # --- label sanity (report only, labels are never modified) ---------------
    labels_clean = df[TARGET_COLUMN].dropna().astype(str)
    n_classes = int(labels_clean.nunique())
    n_classes_normalised = int(labels_clean.str.strip().str.lower().nunique())
    report["n_classes"] = n_classes
    if n_classes_normalised < n_classes:
        warns.append(
            f"{n_classes - n_classes_normalised} label(s) differ only by case/whitespace "
            "(possible duplicate classes). Labels were NOT modified."
        )
    if n_classes < 2:
        critical.append(f"Only {n_classes} distinct class found; need at least 2.")
    if len(feature_cols) == 0:
        critical.append("No feature columns found.")
 
    # --- expected specification ----------------------------------------------
    spec_issues = []
    if len(feature_cols) != EXPECTED_N_FEATURES:
        spec_issues.append(f"features: expected {EXPECTED_N_FEATURES}, found {len(feature_cols)}")
    if n_classes != EXPECTED_N_CLASSES:
        spec_issues.append(f"classes: expected {EXPECTED_N_CLASSES}, found {n_classes}")
    report["spec_discrepancies"] = spec_issues
    if spec_issues:
        message = "Dataset differs from expected specification -> " + "; ".join(spec_issues)
        (warns if allow_spec_mismatch else critical).append(
            message + ("" if allow_spec_mismatch else "  (re-run with --allow-spec-mismatch to proceed)")
        )
 
    if critical:
        print("\n[CRITICAL DATA PROBLEMS] Training stopped:")
        for item in critical:
            print("  x", item)
        raise DataValidationError("; ".join(critical))
 
    # --- build X / y (exact verified spelling) --------------------------------
    X = df.drop(columns=[TARGET_COLUMN]).astype(np.uint8)
    y = df[TARGET_COLUMN].astype(str)
 
    # --- duplicates and conflicting symptom patterns ---------------------------
    pattern_hash = pd.util.hash_pandas_object(X, index=False).values
    frame = pd.DataFrame({"pattern": pattern_hash, "label": y.values})
    dup_rows = int(frame.duplicated().sum())
    n_patterns = int(frame["pattern"].nunique())
    labels_per_pattern = frame.groupby("pattern")["label"].nunique()
    conflicting = labels_per_pattern[labels_per_pattern > 1]
    rows_in_conflict = int(frame["pattern"].isin(conflicting.index).sum())
    pair_counts = frame.groupby(["pattern", "label"]).size()
    accuracy_ceiling = float(pair_counts.groupby(level=0).max().sum() / len(frame))
    report.update(
        duplicate_rows=dup_rows,
        distinct_symptom_patterns=n_patterns,
        conflicting_patterns=int(len(conflicting)),
        rows_in_conflicting_patterns=rows_in_conflict,
        in_sample_accuracy_ceiling=accuracy_ceiling,
    )
    print(f"Exact duplicate rows (features+label): {dup_rows} ({dup_rows / len(frame):.1%})")
    print(f"Distinct symptom patterns: {n_patterns} for {len(frame)} rows")
    print(
        f"Identical symptom patterns with DIFFERENT disease labels: {len(conflicting)} patterns "
        f"({rows_in_conflict} rows, {rows_in_conflict / len(frame):.1%})"
    )
    print(f"In-sample accuracy ceiling implied by label conflicts: {accuracy_ceiling:.1%}")
    if len(conflicting):
        warns.append(
            "Conflicting labels for identical symptom patterns -> some ambiguity cannot be "
            "learned away; top-K metrics are more meaningful than top-1."
        )
        examples = conflicting.sort_values(ascending=False).head(3).index
        for pattern in examples:
            labels = frame.loc[frame["pattern"] == pattern, "label"].unique()
            print(f"   e.g. one pattern -> {len(labels)} labels: {list(labels[:4])}")
    if dup_rows:
        warns.append(
            "Duplicate rows can leak between train and test; see the 'novel pattern' test metrics."
        )
 
    for item in warns:
        print("  ! WARNING:", item)
    report["warnings"] = warns
 
    print("\nFeature shape:", X.shape)
    print("Target shape:", y.shape)
    print("Unique classes:", y.nunique())
    return X, y, report
 
 
# --------------------------------------------------------------------------- #
# STEP 4 - Class distribution
# --------------------------------------------------------------------------- #
def analyze_class_distribution(y: pd.Series, reports_dir: Path, make_plots: bool = False):
    counts = y.value_counts()
    summary = {
        "n_classes": int(len(counts)),
        "min_samples": int(counts.min()),
        "max_samples": int(counts.max()),
        "median_samples": float(counts.median()),
        "imbalance_ratio_max_over_min": float(counts.max() / counts.min()),
        "classes_with_1_sample": int((counts == 1).sum()),
        "classes_lt_5_samples": int((counts < 5).sum()),
        "classes_lt_10_samples": int((counts < 10).sum()),
        "classes_lt_20_samples": int((counts < 20).sum()),
    }
    print(counts.describe().round(2).to_string())
    print("\nMost frequent diseases:\n" + counts.head(10).to_string())
    print("\nLeast frequent diseases:\n" + counts.tail(10).to_string())
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(
        "\nWhy this matters: with this many classes a few common diseases can dominate "
        "accuracy, rare classes get little signal (low recall), and singleton classes "
        "cannot be evaluated independently. Read macro-F1 / balanced accuracy / "
        "per-class recall, not accuracy alone."
    )
    counts.rename_axis("disease").rename("n_samples").to_csv(reports_dir / "class_distribution.csv")
 
    if make_plots:
        try:
            import matplotlib
 
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
 
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(counts.values, bins=50)
            ax.set_yscale("log")
            ax.set_xlabel("Samples per disease")
            ax.set_ylabel("Number of diseases (log)")
            ax.set_title("Class frequency distribution")
            fig.tight_layout()
            fig.savefig(reports_dir / "class_distribution.png", dpi=120)
            plt.close(fig)
        except ImportError:
            print("matplotlib not installed - skipping plot.")
    return counts, summary
 
 
# --------------------------------------------------------------------------- #
# STEP 5 - Splitting (with explicit rare-class handling)
# --------------------------------------------------------------------------- #
def stratified_split_with_rare_handling(X, y, test_size, random_state, stage):
    """
    Stratified split that stays honest about rare classes.
 
    * Classes with < 2 samples cannot appear in both partitions. They are kept in
      the TRAIN partition only (never removed, never duplicated) and reported as
      not independently evaluable.
    * If the remaining data still cannot be stratified (e.g. the test partition
      would be smaller than the number of classes) a plain random split is used
      and the reason is reported.
    """
    counts = y.value_counts()
    rare_classes = counts[counts < 2].index.tolist()
    rare_mask = y.isin(rare_classes)
    X_main, y_main = X.loc[~rare_mask], y.loc[~rare_mask]
    if len(y_main) < 2:
        raise DataValidationError(f"[{stage}] Not enough data to split.")
 
    n_classes_main = int(y_main.nunique())
    n_test = int(np.ceil(test_size * len(y_main)))
    n_train = len(y_main) - n_test
    reasons = []
    if n_test < n_classes_main:
        reasons.append(f"test partition ({n_test} rows) is smaller than the number of classes ({n_classes_main})")
    if n_train < n_classes_main:
        reasons.append(f"train partition ({n_train} rows) is smaller than the number of classes ({n_classes_main})")
    stratified = not reasons
 
    if rare_classes:
        print(
            f"[{stage}] {len(rare_classes)} class(es) have a single sample and cannot be stratified: "
            f"{rare_classes[:20]}{' ...' if len(rare_classes) > 20 else ''}\n"
            f"[{stage}] They are kept in TRAIN only (not removed, not duplicated); "
            "they cannot be evaluated on independent data without more records."
        )
    if not stratified:
        print(f"[{stage}] Stratification NOT possible: " + "; ".join(reasons) + ". Using a random split.")
 
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_main, y_main, test_size=test_size, random_state=random_state,
        stratify=y_main if stratified else None,
    )
    if rare_classes:
        X_tr = pd.concat([X_tr, X.loc[rare_mask]]).sort_index()
        y_tr = pd.concat([y_tr, y.loc[rare_mask]]).sort_index()
    X_te, y_te = X_te.sort_index(), y_te.sort_index()
 
    missing_in_train = sorted(set(y_te.unique()) - set(y_tr.unique()))
    if missing_in_train:
        print(f"[{stage}] WARNING: {len(missing_in_train)} class(es) appear in test but not in train.")
    info = {
        "stage": stage,
        "test_size": test_size,
        "stratified": stratified,
        "reasons_not_stratified": reasons,
        "rare_classes_train_only": rare_classes,
        "classes_in_test_missing_from_train": missing_in_train,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }
    return X_tr, X_te, y_tr, y_te, info
 
 
def split_data(X, y, test_size=0.20, random_state=42):
    X_train, X_test, y_train, y_test, info = stratified_split_with_rare_handling(
        X, y, test_size, random_state, stage="train/test"
    )
    train_counts = y_train.value_counts()
    info["classes_lt_5_train_samples"] = int((train_counts < 5).sum())
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | stratified: {info['stratified']}")
    print(f"Classes in train: {y_train.nunique()} | in test: {y_test.nunique()} | "
          f"classes with <5 train samples: {info['classes_lt_5_train_samples']}")
    print("The test set is now frozen - it is NOT used for model selection or tuning.")
    return X_train, X_test, y_train, y_test, info
 
 
# --------------------------------------------------------------------------- #
# Symptom dropout
# --------------------------------------------------------------------------- #
def random_symptom_subset(values: np.ndarray, keep_counts, rng) -> sparse.csr_matrix:
    """Keep a random keep_counts symptoms per row (all of them if a row has fewer)."""
    rows, cols = np.nonzero(values)
    order = np.lexsort((rng.random(len(rows)), rows))
    rows, cols = rows[order], cols[order]
    row_sizes = np.bincount(rows, minlength=values.shape[0])
    row_starts = np.cumsum(row_sizes) - row_sizes
    rank_in_row = np.arange(len(rows)) - row_starts[rows]
    limits = np.broadcast_to(keep_counts, (values.shape[0],))
    kept = rank_in_row < limits[rows]
    data = np.ones(int(kept.sum()), dtype=np.float32)
    return sparse.csr_matrix((data, (rows[kept], cols[kept])), shape=values.shape, dtype=np.float32)


def add_symptom_dropout(X: pd.DataFrame, y: pd.Series, copies: int, random_state: int):
    """
    Stack the rows with copies that keep only a few symptoms each. Dataset rows have about
    5 symptoms but users often type 1 to 3, and a tree trained only on full rows fails on
    those. Without one-symptom copies it was right 5% of the time while claiming 57%. Sparse output keeps memory small. Apply to training data only, after splitting.
    """
    rng = np.random.default_rng(random_state)
    values = X.to_numpy()
    low, high = DROPOUT_KEEP_RANGE
    blocks = [sparse.csr_matrix(values, dtype=np.float32)]
    for _ in range(copies):
        blocks.append(random_symptom_subset(values, rng.integers(low, high + 1, size=len(values)), rng))
    return sparse.vstack(blocks, format="csc"), np.tile(np.asarray(y), copies + 1)


def keep_few_symptoms(X: pd.DataFrame, k: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    reduced = random_symptom_subset(X.to_numpy(), k, rng).toarray().astype(np.uint8)
    return pd.DataFrame(reduced, columns=X.columns, index=X.index)


def fit_on_dropout_data(estimator, X_sparse, labels, feature_names):
    estimator.fit(X_sparse, labels)
    # Sparse input carries no column names. Recording them makes sklearn reject any
    # later DataFrame whose columns are in a different order.
    estimator.feature_names_in_ = np.asarray(feature_names, dtype=object)
    return estimator


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
 
 
def get_scores(model, X):
    """Ranking scores for top-K. Returns (scores, kind) or (None, None). NOT calibrated."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X), "predict_proba"
    if hasattr(model, "decision_function"):
        return model.decision_function(X), "decision_function"
    return None, None
 
 
def score_kind(model) -> str | None:
    if hasattr(model, "predict_proba"):
        return "predict_proba (uncalibrated)"
    if hasattr(model, "decision_function"):
        return "decision_function (uncalibrated margin)"
    return None
 
 
def top_k_accuracy(model, X, y_true, ks=TOP_K_VALUES, batch_size=5000):
    """Top-K accuracy computed in batches. Ranking quality only - not diagnostic certainty."""
    classes = np.asarray(model.classes_)
    if get_scores(model, X.iloc[:1])[0] is None:
        return None
    k_max = min(max(ks), len(classes))
    y_arr = np.asarray(y_true)
    hits = {k: 0 for k in ks}
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start:start + batch_size]
        scores, _ = get_scores(model, batch)
        part = np.argpartition(-scores, k_max - 1, axis=1)[:, :k_max]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        top_idx = np.take_along_axis(part, order, axis=1)
        match = classes[top_idx] == y_arr[start:start + len(batch), None]
        for k in ks:
            hits[k] += int(match[:, :min(k, k_max)].any(axis=1).sum())
    return {f"top_{k}_accuracy": hits[k] / len(X) for k in ks}


def mean_top_score(model, X, batch_size=5000) -> float:
    """Compare with top_1_accuracy: a much higher score means overconfidence."""
    top_scores = [model.predict_proba(X.iloc[start:start + batch_size]).max(axis=1).sum()
                  for start in range(0, len(X), batch_size)]
    return float(sum(top_scores) / len(X))
 
 
def few_symptom_metrics(model, X, y_true, random_state, batch_size=5000) -> dict:
    """Ranking quality when only a few symptoms are given, as with free-text input."""
    results = {}
    for k in FEW_SYMPTOM_COUNTS:
        reduced = keep_few_symptoms(X, k, random_state)
        row = top_k_accuracy(model, reduced, y_true) or {}
        if hasattr(model, "predict_proba"):
            row["mean_top_score"] = mean_top_score(model, reduced, batch_size)
        results[f"{k}_symptoms"] = row
    return results

 
# --------------------------------------------------------------------------- #
# STEP 6/7 - Candidate comparison, tuning, final fit
# --------------------------------------------------------------------------- #
def train_models(X_train, y_train, model_names, val_size, random_state, n_jobs, tuning_rows=0,
                 dropout_copies=0):
    """
    Compare candidates + small grids on a VALIDATION split of the training data,
    pick the best by the mean macro-F1 on full and few-symptom rows, then refit it on the
    full training set. The test set is never touched here.
    """
    feature_names = list(X_train.columns)
    X_fit, X_val, y_fit, y_val, val_info = stratified_split_with_rare_handling(
        X_train, y_train, val_size, random_state, stage="train/validation"
    )
    if tuning_rows and len(X_fit) > tuning_rows:
        sample_idx = X_fit.sample(n=tuning_rows, random_state=random_state).index
        X_fit, y_fit = X_fit.loc[sample_idx], y_fit.loc[sample_idx]
        print(f"Tuning on a random subsample of {tuning_rows} fit rows (--tuning-rows).")
    print(f"Fit: {X_fit.shape} | Validation: {X_val.shape}")
    X_fit_aug, y_fit_aug = add_symptom_dropout(X_fit, y_fit, dropout_copies, random_state)
    X_val_few = keep_few_symptoms(X_val, SELECTION_FEW_SYMPTOMS, random_state)
    print(f"Fit rows with {dropout_copies} symptom-dropout copies: {X_fit_aug.shape}")

    results = []
    for name in model_names:
        factory, grid = MODEL_REGISTRY[name]
        for params in ParameterGrid(grid):
            estimator = factory(random_state, n_jobs).set_params(**params)
            print(f"-> fitting {name} {params} ...", flush=True)
            start = time.perf_counter()
            try:
                fit_on_dropout_data(estimator, X_fit_aug, y_fit_aug, feature_names)
            except MemoryError:
                print(f"   skipped (MemoryError). Try --tuning-rows or drop {name}.")
                continue
            fit_seconds = time.perf_counter() - start
            metrics = compute_metrics(y_val, estimator.predict(X_val))
            few_f1 = compute_metrics(y_val, estimator.predict(X_val_few))["macro_f1"]
            results.append({"model": name, "params": params, "fit_seconds": round(fit_seconds, 2),
                            "selection_score": (metrics["macro_f1"] + few_f1) / 2,
                            "few_symptom_macro_f1": few_f1, **metrics})
            print(f"   macro_f1={metrics['macro_f1']:.4f} few_symptom_macro_f1={few_f1:.4f} "
                  f"bal_acc={metrics['balanced_accuracy']:.4f} "
                  f"acc={metrics['accuracy']:.4f} ({fit_seconds:.1f}s)")
            del estimator
 
    if not results:
        raise RuntimeError("No candidate model could be trained successfully.")
 
    ranked = sorted(
        results, key=lambda r: (r["selection_score"], r["balanced_accuracy"], r["accuracy"]), reverse=True
    )
    table = pd.DataFrame(ranked)
    table["params"] = table["params"].astype(str)
    print("\nValidation leaderboard (selection metric: mean of full and few-symptom macro-F1):")
    print(table.round(4).to_string(index=False))
 
    best = ranked[0]
    print(f"\nSelected: {best['model']} {best['params']}")
    factory, _ = MODEL_REGISTRY[best["model"]]
    final_model = factory(random_state, n_jobs).set_params(**best["params"])
    print("Refitting selected configuration on the FULL training set ...")
    del X_fit_aug, y_fit_aug
    X_train_aug, y_train_aug = add_symptom_dropout(X_train, y_train, dropout_copies, random_state)
    start = time.perf_counter()
    fit_on_dropout_data(final_model, X_train_aug, y_train_aug, feature_names)
    final_fit_seconds = time.perf_counter() - start
    print(f"Final fit done in {final_fit_seconds:.1f}s")
 
    if set(final_model.classes_) != set(y_train.unique()):
        print("WARNING: model.classes_ does not match the training labels.")
 
    selection = {
        "selected_model": best["model"],
        "selected_params": best["params"],
        "best_validation_metrics": {k: best[k] for k in
                                    ("accuracy", "balanced_accuracy", "macro_precision",
                                     "macro_recall", "macro_f1", "weighted_f1")},
        "leaderboard": ranked,
        "validation_split": val_info,
        "final_fit_seconds": round(final_fit_seconds, 2),
        "tuning_rows_cap": tuning_rows or None,
    }
    return final_model, selection
 
 
# --------------------------------------------------------------------------- #
# STEP 8 - Evaluation on the untouched test set
# --------------------------------------------------------------------------- #
def evaluate_model(model, X_train, y_train, X_test, y_test, reports_dir: Path, random_state=42):
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print("Overall test metrics:")
    for key, value in metrics.items():
        print(f"  {key:<18}: {value:.4f}")
 
    top_k = top_k_accuracy(model, X_test, y_test)
    if top_k:
        print("Top-K (ranking quality only, NOT diagnostic certainty):")
        for key, value in top_k.items():
            print(f"  {key:<18}: {value:.4f}")

    few_symptoms = few_symptom_metrics(model, X_test, y_test, random_state)
    print("Few-symptom inputs (random symptoms kept per test row, as with free text):")
    for key, row in few_symptoms.items():
        print(f"  {key:<18}: " + " | ".join(f"{name}={value:.4f}" for name, value in row.items()))

    # Full per-class report -> files (773 lines is too long to print)
    (reports_dir / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, zero_division=0)
    )
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    per_class = pd.DataFrame(report_dict).T.drop(
        index=[r for r in ("accuracy", "micro avg", "macro avg", "weighted avg") if r in report_dict]
    )
    per_class.index.name = "disease"
    per_class.to_csv(reports_dir / "per_class_metrics.csv")
 
    scored = per_class[per_class["support"] > 0]
    print(f"\nClasses evaluated: {len(scored)} | recall==0: {int((scored['recall'] == 0).sum())} | "
          f"recall==1: {int((scored['recall'] == 1).sum())} | "
          f"support<5: {int((scored['support'] < 5).sum())}")
    cols = ["precision", "recall", "f1-score", "support"]
    print("\nLowest-recall classes:\n" + scored.sort_values(["recall", "support"]).head(10)[cols].round(3).to_string())
    print("\nLowest-precision classes (that were predicted):\n"
          + per_class[per_class["precision"] > 0].sort_values("precision").head(10)[cols].round(3).to_string())
 
    # Are common diseases dominating?
    train_counts = y_train.value_counts()
    freq = scored.copy()
    freq["train_count"] = freq.index.map(train_counts).fillna(0)
    try:
        freq["train_freq_quartile"] = pd.qcut(freq["train_count"], q=4, duplicates="drop")
        bucket = freq.groupby("train_freq_quartile", observed=True).agg(
            n_classes=("recall", "size"), mean_recall=("recall", "mean"), mean_f1=("f1-score", "mean")
        )
        print("\nPerformance by training-frequency quartile (rare -> common):\n" + bucket.round(3).to_string())
    except ValueError:
        pass
    print(f"\nAccuracy {metrics['accuracy']:.3f} vs macro-F1 {metrics['macro_f1']:.3f}: "
          "a large gap means common diseases are carrying the headline number.")
 
    # Most frequent confusions (instead of a 773x773 matrix)
    wrong = pd.DataFrame({"true": y_test.values, "pred": y_pred})
    wrong = wrong[wrong["true"] != wrong["pred"]]
    pairs = wrong.groupby(["true", "pred"]).size().sort_values(ascending=False).rename("count").reset_index()
    pairs.head(500).to_csv(reports_dir / "top_confusions.csv", index=False)
    print("\nMost frequent confusions (true -> predicted):\n" + pairs.head(10).to_string(index=False))
 
    # Leakage-aware view: test rows whose exact symptom pattern never appeared in train
    train_hashes = pd.util.hash_pandas_object(X_train, index=False).values
    test_hashes = pd.util.hash_pandas_object(X_test, index=False).values
    seen = np.isin(test_hashes, train_hashes)
    novel_metrics, novel_n = None, int((~seen).sum())
    print(f"\nTest rows whose symptom pattern also exists in train: {seen.mean():.1%}")
    if novel_n >= 30:
        novel_metrics = compute_metrics(y_test.values[~seen], y_pred[~seen])
        print(f"Metrics on the {novel_n} NOVEL-pattern test rows (more honest generalization estimate):")
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
            print(f"  {key:<18}: {novel_metrics[key]:.4f}")
        if top_k:
            novel_top_k = top_k_accuracy(model, X_test.loc[~seen], y_test.loc[~seen])
            print("  " + " | ".join(f"{k}={v:.4f}" for k, v in novel_top_k.items()))
            novel_metrics.update(novel_top_k)
    else:
        print("Too few novel-pattern rows for a separate estimate.")
 
    evaluation = {
        "test_metrics": metrics,
        "top_k": top_k,
        "few_symptoms": few_symptoms,
        "score_type": score_kind(model),
        "n_test_rows": int(len(y_test)),
        "n_classes_in_test": int(y_test.nunique()),
        "classes_with_test_support_lt_5": int((scored["support"] < 5).sum()),
        "share_test_rows_with_pattern_seen_in_train": float(seen.mean()),
        "novel_pattern_test_rows": novel_n,
        "novel_pattern_metrics": novel_metrics,
    }
    return evaluation
 
 
# --------------------------------------------------------------------------- #
# STEP 9 - Overfitting / generalization check
# --------------------------------------------------------------------------- #
def check_generalization(model, X_train, y_train, selection, test_metrics, random_state, sample_size=20000):
    sample_idx = X_train.sample(n=min(sample_size, len(X_train)), random_state=random_state).index
    train_metrics = compute_metrics(y_train.loc[sample_idx], model.predict(X_train.loc[sample_idx]))
    rows = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    table = pd.DataFrame({
        "train(sample)": train_metrics,
        "validation": selection["best_validation_metrics"],
        "test": test_metrics,
    }).loc[rows]
    print(table.round(4).to_string())
 
    findings = []
    gap_f1 = train_metrics["macro_f1"] - test_metrics["macro_f1"]
    gap_acc = train_metrics["accuracy"] - test_metrics["accuracy"]
    if gap_f1 > 0.10 or gap_acc > 0.10:
        findings.append(
            f"Possible overfitting (train-test gap: acc {gap_acc:.2f}, macro-F1 {gap_f1:.2f}). "
            "Consider stronger regularisation, simpler models, or more data for rare classes."
        )
    if test_metrics["macro_f1"] < 0.30:
        findings.append("Low macro-F1: possible underfitting, label ambiguity, or too few samples per class.")
    if test_metrics["accuracy"] - test_metrics["macro_f1"] > 0.20:
        findings.append("Accuracy >> macro-F1: performance is concentrated in common diseases.")
    if abs(selection["best_validation_metrics"]["macro_f1"] - test_metrics["macro_f1"]) > 0.05:
        findings.append("Validation and test macro-F1 differ noticeably; the split may be too small/noisy.")
    if not findings:
        findings.append("No major red flag from this check - still validate on external data.")
    for finding in findings:
        print(" *", finding)
    return {"train_sample_metrics": train_metrics, "findings": findings}
 
 
# --------------------------------------------------------------------------- #
# STEP 10 - Model comparison + export
# --------------------------------------------------------------------------- #
def compare_models(model, selection, evaluation, X_train, y_train, X_test, y_test,
                   dropout_copies, random_state, reports_dir: Path) -> list:
    """
    Reports the exported model against Naive Bayes on the same training and test data.
    This never changes which model is exported: the tree is kept because every prediction
    can be traced to its splits, which Debug Mode and the follow-up questions rely on.
    alpha is fixed rather than tuned, using a value that did well in earlier experiments.
    """
    X_aug, y_aug = add_symptom_dropout(X_train, y_train, dropout_copies, random_state)
    baseline = BernoulliNB(alpha=0.01, binarize=None)
    start = time.perf_counter()
    fit_on_dropout_data(baseline, X_aug, y_aug, list(X_train.columns))
    baseline_seconds = time.perf_counter() - start

    def rows_for(name, explainable, fit_seconds, macro_f1, all_symptoms, few_symptoms, mean_score):
        rows = [{"model": name, "symptoms_given": "all", **all_symptoms, "mean_top_score": mean_score,
                 "macro_f1": macro_f1, "fit_seconds": round(fit_seconds, 1), "explainable": explainable}]
        for key, metrics in few_symptoms.items():
            rows.append({"model": name, "symptoms_given": key.split("_")[0], **metrics, "macro_f1": None,
                         "fit_seconds": round(fit_seconds, 1), "explainable": explainable})
        return rows

    rows = rows_for(type(model).__name__, hasattr(model, "tree_"), selection["final_fit_seconds"],
                    evaluation["test_metrics"]["macro_f1"], evaluation["top_k"], evaluation["few_symptoms"],
                    mean_top_score(model, X_test))
    rows += rows_for(type(baseline).__name__, False, baseline_seconds,
                     compute_metrics(y_test, baseline.predict(X_test))["macro_f1"],
                     top_k_accuracy(baseline, X_test, y_test),
                     few_symptom_metrics(baseline, X_test, y_test, random_state),
                     mean_top_score(baseline, X_test))

    table = pd.DataFrame(rows)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    print(table.round(3).to_string(index=False))
    return rows
 
 
def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
 
 
def save_model(model, feature_names, metadata_extras: dict, output_dir: Path):
    model_path = output_dir / MODEL_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    joblib.dump(model, model_path, compress=3)  # tree value arrays are mostly zeros

    class_labels = [str(c) for c in model.classes_]
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "development prototype - not clinically validated",
        "disclaimer": DISCLAIMER,
        "target_column": TARGET_COLUMN,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),  # EXACT training order
        "input_spec": {"dtype": "binary 0/1", "order": "same as feature_names", "shape": [1, len(feature_names)]},
        "n_classes": len(class_labels),
        "class_labels": class_labels,  # same order as model.classes_ / score columns
        "selected_algorithm": type(model).__name__,
        "score_type": score_kind(model),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        **metadata_extras,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved model    -> {model_path} ({model_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved metadata -> {metadata_path}")
    # Written from the training columns on every run, so it can't drift from the model
    schema_path = output_dir / SCHEMA_FILENAME
    schema_path.write_text(json.dumps(list(feature_names), indent=2) + "\n", encoding="utf-8")
    print(f"Saved schema   -> {schema_path}")
    return model_path, metadata_path
 
 
# --------------------------------------------------------------------------- #
# STEP 11 - Backend-facing helpers (input validation + prediction interface)
# --------------------------------------------------------------------------- #
def validate_symptom_input(symptoms: dict, feature_names, missing_as_zero: bool = True) -> pd.DataFrame:
    """
    Strict validation: only known symptom names, values must be 0/1, exact feature
    order. Symptoms not supplied are treated as 0 ("not reported") unless
    missing_as_zero=False.
    """
    unknown = sorted(set(symptoms) - set(feature_names))
    if unknown:
        raise ValueError(f"Unknown symptom name(s): {unknown[:10]}")
    if not missing_as_zero:
        missing = [n for n in feature_names if n not in symptoms]
        if missing:
            raise ValueError(f"Missing symptom value(s): {missing[:10]}")
    row = []
    for name in feature_names:
        value = symptoms.get(name, 0)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, str) or value not in (0, 1):
            raise ValueError(f"Symptom '{name}' must be 0 or 1, got {value!r}")
        row.append(int(value))
    return pd.DataFrame([row], columns=list(feature_names), dtype=np.uint8)
 
 
def predict_top_k(model, metadata: dict, symptoms: dict, k: int = 5, missing_as_zero: bool = True) -> dict:
    """Reference prediction interface for the future backend API."""
    frame = validate_symptom_input(symptoms, metadata["feature_names"], missing_as_zero)
    scores, kind = get_scores(model, frame)
    classes = np.asarray(model.classes_)
    if scores is None:
        return {"predicted_disease": str(model.predict(frame)[0]), "top_k": [],
                "score_type": None, "model_version": metadata["model_version"], "disclaimer": DISCLAIMER}
    order = np.argsort(-scores[0])[: min(k, len(classes))]
    return {
        "predicted_disease": str(classes[order[0]]),
        "top_k": [{"disease": str(classes[i]), "score": float(scores[0][i])} for i in order],
        "score_type": f"{kind} (uncalibrated - not a medical confidence value)",
        "model_version": metadata["model_version"],
        "disclaimer": DISCLAIMER,
    }
 
 
def smoke_test_export(model_path: Path, metadata_path: Path, X_test: pd.DataFrame, n_rows: int = 3) -> bool:
    """Reload the saved artifacts exactly as a backend would and sanity-check them."""
    loaded = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ok = metadata["feature_names"] == list(X_test.columns)
    if not ok:
        print("SMOKE TEST FAILED: metadata feature order differs from the data.")
        return False
    for i in range(min(n_rows, len(X_test))):
        row = X_test.iloc[i]
        payload = {name: int(row[name]) for name in metadata["feature_names"]}
        result = predict_top_k(loaded, metadata, payload, k=3)
        direct = str(loaded.predict(X_test.iloc[[i]])[0])
        if result["predicted_disease"] != direct:
            print(f"  note: row {i}: score-argmax '{result['predicted_disease']}' != predict() '{direct}' (tie?)")
    print("Export smoke test passed (artifact reloads; predict_top_k works).")
    return True
 
 
# --------------------------------------------------------------------------- #
# CLI + main
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train the Sanjeevani AI multiclass model (prototype).")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default="models", help="Where the model + metadata are written.")
    parser.add_argument("--reports-dir", default="reports", help="Where CSV/TXT reports are written.")
    parser.add_argument("--models", default=DEFAULT_MODELS,
                        help=f"Comma-separated candidates from: {', '.join(MODEL_REGISTRY)}")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation share carved out of TRAIN.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Used by random_forest only.")
    parser.add_argument("--tuning-rows", type=int, default=0,
                        help="Cap rows used for candidate comparison (0 = use all). Speeds up slow models.")
    parser.add_argument("--dropout-copies", type=int, default=6,
                        help="Extra training copies of each row with only a few symptoms kept (0 = none).")
    parser.add_argument("--allow-spec-mismatch", action="store_true",
                        help="Proceed even if feature/class counts differ from 377/773.")
    parser.add_argument("--plots", action="store_true", help="Save a class-distribution plot (needs matplotlib).")
    args = parser.parse_args()
    args.model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in args.model_list if m not in MODEL_REGISTRY]
    if unknown or not args.model_list:
        parser.error(f"Unknown/empty --models {unknown}. Choose from: {list(MODEL_REGISTRY)}")
    return args
 
 
def main():
    args = parse_args()
    np.random.seed(args.random_state)
    output_dir, reports_dir = Path(args.output_dir), Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
 
    print("Sanjeevani AI - model training (DEVELOPMENT PROTOTYPE, not clinically validated)")
    try:
        section("STEP 1 - DATA LOADING")
        df = load_data(args.data)
        dataset_sha256 = file_sha256(args.data)
 
        section("STEP 2/3 - DATA VALIDATION & FEATURE-TARGET SPLIT")
        X, y, validation_report = validate_data(df, args.allow_spec_mismatch)
        del df
        feature_names = list(X.columns)
 
        section("STEP 4 - CLASS DISTRIBUTION & IMBALANCE")
        _, class_summary = analyze_class_distribution(y, reports_dir, args.plots)
 
        section("STEP 5 - TRAIN/TEST SPLIT")
        X_train, X_test, y_train, y_test, split_info = split_data(X, y, args.test_size, args.random_state)
 
        section("STEP 6/7 - MODEL COMPARISON, TUNING & FINAL FIT (validation split only)")
        model, selection = train_models(
            X_train, y_train, args.model_list, args.val_size,
            args.random_state, args.n_jobs, args.tuning_rows, args.dropout_copies,
        )
 
        section("STEP 8 - TEST-SET EVALUATION (untouched holdout)")
        evaluation = evaluate_model(model, X_train, y_train, X_test, y_test, reports_dir, args.random_state)
 
        section("STEP 9 - OVERFITTING & GENERALIZATION CHECK")
        generalization = check_generalization(
            model, X_train, y_train, selection, evaluation["test_metrics"], args.random_state
        )
 
        section("STEP 10 - MODEL COMPARISON & EXPORT")
        comparison = compare_models(
            model, selection, evaluation, X_train, y_train, X_test, y_test,
            args.dropout_copies, args.random_state, reports_dir,
        )
        extras = {
            "dataset": {"path": args.data, "sha256": dataset_sha256},
            "training_config": {
                "random_state": args.random_state,
                "test_size": args.test_size,
                "val_size": args.val_size,
                "candidates": args.model_list,
                "tuning_rows_cap": args.tuning_rows or None,
                "dropout_copies": args.dropout_copies,
                "split": split_info,
            },
            "model_selection": selection,
            "data_validation": validation_report,
            "class_distribution": class_summary,
            "evaluation": evaluation,
            "generalization": generalization,
            "model_comparison": comparison,
        }
        model_path, metadata_path = save_model(model, feature_names, extras, output_dir)
        smoke_test_export(model_path, metadata_path, X_test)
 
        section("DONE")
        print(f"Selected: {selection['selected_model']} {selection['selected_params']}")
        print(f"Test macro-F1: {evaluation['test_metrics']['macro_f1']:.4f} | "
              f"accuracy: {evaluation['test_metrics']['accuracy']:.4f}")
        print("Reminder: prototype only. Not a diagnosis; scores are uncalibrated; "
              "clinician escalation and emergency detection must be handled separately.")
 
    except (DataValidationError, FileNotFoundError) as exc:
        print(f"\n[STOPPED] {exc}", file=sys.stderr)
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()
 