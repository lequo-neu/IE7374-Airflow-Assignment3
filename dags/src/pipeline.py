"""
Customer segmentation pipeline functions used by the Airflow DAG.

This module is intentionally free of Airflow imports so every function
can be unit tested with plain pytest, without a running Airflow instance.

Data is passed between tasks through XCom as base64-encoded pickles,
which keeps every return value JSON-safe (same approach as the original lab).
"""

import base64
import json
import os
import pickle

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "file.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "kmeans_bundle.pkl")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

FEATURES = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]
K_RANGE = list(range(2, 16))
KMEANS_KWARGS = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 42}


# ----------------- Serialization helpers ----------------- #
def encode(obj) -> str:
    """Pickle an object and return it as a base64 string (JSON-safe for XCom)."""
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def decode(data_b64: str):
    """Reverse of encode(): base64 string back to the original Python object."""
    return pickle.loads(base64.b64decode(data_b64))


# ----------------- Task 1: Load ----------------- #
def load_data(csv_path: str = DATA_PATH) -> str:
    """
    Load the raw credit card customer CSV.

    Args:
        csv_path (str): Path to the CSV file.
    Returns:
        str: base64-encoded pickled DataFrame.
    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from {csv_path}")
    return encode(df)


# ----------------- Task 2: Validate ----------------- #
def validate_data(data_b64: str, min_rows: int = 1000, max_null_ratio: float = 0.05) -> dict:
    """
    Check that the data is fit for clustering before any compute is spent.

    Args:
        data_b64 (str): Output of load_data().
        min_rows (int): Minimum number of rows required.
        max_null_ratio (float): Maximum allowed share of nulls per feature.
    Returns:
        dict: Validation summary (row count and null ratio per feature).
    Raises:
        ValueError: If a feature is missing, there are too few rows,
            or a feature has too many nulls.
    """
    df = decode(data_b64)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < min_rows:
        raise ValueError(f"Only {len(df)} rows, need at least {min_rows}")

    null_ratio = {c: round(float(df[c].isna().mean()), 4) for c in FEATURES}
    too_many = {c: r for c, r in null_ratio.items() if r > max_null_ratio}
    if too_many:
        raise ValueError(f"Null ratio above {max_null_ratio}: {too_many}")

    summary = {"rows": int(len(df)), "null_ratio": null_ratio}
    print(f"Validation passed: {summary}")
    return summary


# ----------------- Task 3: Preprocess ----------------- #
def preprocess_data(data_b64: str) -> str:
    """
    Select features, drop nulls, log-transform skewed amounts, and standardize.

    Args:
        data_b64 (str): Output of load_data().
    Returns:
        str: base64-encoded pickle of (X_scaled, fitted_scaler).
    """
    df = decode(data_b64)
    features = df[FEATURES].dropna()
    X_log = np.log1p(features.values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    print(f"Preprocessed matrix shape: {X_scaled.shape}")
    return encode((X_scaled, scaler))


# ----------------- Task 4a: Elbow (parallel) ----------------- #
def compute_elbow(prep_b64: str) -> dict:
    """
    Fit KMeans for every k in K_RANGE and find the elbow of the SSE curve.

    Args:
        prep_b64 (str): Output of preprocess_data().
    Returns:
        dict: {"k_elbow": int, "sse": list of float}
    """
    X, _ = decode(prep_b64)
    sse = [float(KMeans(n_clusters=k, **KMEANS_KWARGS).fit(X).inertia_) for k in K_RANGE]
    knee = KneeLocator(K_RANGE, sse, curve="convex", direction="decreasing").elbow
    k_elbow = int(knee) if knee is not None else K_RANGE[int(np.argmax(np.diff(sse, 2))) + 1]
    print(f"Elbow method suggests k = {k_elbow}")
    return {"k_elbow": k_elbow, "sse": [round(s, 2) for s in sse]}


# ----------------- Task 4b: Silhouette (parallel) ----------------- #
def compute_silhouette(prep_b64: str, sample_size: int = 2000) -> dict:
    """
    Score every k in K_RANGE with the silhouette coefficient.

    Args:
        prep_b64 (str): Output of preprocess_data().
        sample_size (int): Rows sampled per score to keep runtime low.
    Returns:
        dict: {"k_silhouette": int, "scores": {str(k): float}}
    """
    X, _ = decode(prep_b64)
    scores = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, **KMEANS_KWARGS).fit_predict(X)
        scores[str(k)] = round(
            float(silhouette_score(X, labels, sample_size=sample_size, random_state=42)), 4
        )
    k_best = int(max(scores, key=scores.get))
    print(f"Silhouette method suggests k = {k_best} (score {scores[str(k_best)]})")
    return {"k_silhouette": k_best, "scores": scores}


# ----------------- Task 5: Select k (fan-in) ----------------- #
def select_k(elbow_result: dict, silhouette_result: dict, tolerance: float = 0.05) -> dict:
    """
    Combine both methods into one decision.

    Keep the elbow k if its silhouette is within `tolerance` of the best
    silhouette; otherwise switch to the silhouette winner.

    Args:
        elbow_result (dict): Output of compute_elbow().
        silhouette_result (dict): Output of compute_silhouette().
        tolerance (float): Allowed silhouette gap before overriding the elbow.
    Returns:
        dict: chosen_k, silhouette at chosen_k, both candidates, and the reason.
    Raises:
        ValueError: If the elbow k has no silhouette score.
    """
    k_elbow = elbow_result["k_elbow"]
    k_sil = silhouette_result["k_silhouette"]
    scores = silhouette_result["scores"]
    if str(k_elbow) not in scores:
        raise ValueError(f"No silhouette score for elbow k = {k_elbow}")

    best = scores[str(k_sil)]
    if scores[str(k_elbow)] >= best - tolerance:
        chosen, reason = k_elbow, "elbow k kept (silhouette within tolerance)"
    else:
        chosen, reason = k_sil, "elbow k overridden by silhouette winner"

    selection = {
        "chosen_k": chosen,
        "silhouette": scores[str(chosen)],
        "k_elbow": k_elbow,
        "k_silhouette": k_sil,
        "reason": reason,
    }
    print(f"Model selection: {selection}")
    return selection


# ----------------- Task 6: Quality gate (branch) ----------------- #
def quality_gate(selection: dict, threshold: float = 0.25) -> str:
    """
    Decide which branch the DAG should follow.

    Args:
        selection (dict): Output of select_k().
        threshold (float): Minimum silhouette to accept the clustering.
    Returns:
        str: Task id of the next task to run.
    """
    if selection["silhouette"] >= threshold:
        print(f"Quality gate PASSED: {selection['silhouette']} >= {threshold}")
        return "train_final_model_task"
    print(f"Quality gate FAILED: {selection['silhouette']} < {threshold}")
    return "quality_alert_task"


# ----------------- Task 7: Train final model ----------------- #
def train_final_model(prep_b64: str, selection: dict, model_path: str = MODEL_PATH) -> dict:
    """
    Retrain KMeans with the chosen k and save it together with its scaler.

    Args:
        prep_b64 (str): Output of preprocess_data().
        selection (dict): Output of select_k().
        model_path (str): Where to save the model bundle.
    Returns:
        dict: chosen k, cluster sizes, and saved model path.
    """
    X, scaler = decode(prep_b64)
    k = selection["chosen_k"]
    model = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(X)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "k": k, "features": FEATURES}, f)

    sizes = {str(c): int(n) for c, n in zip(*np.unique(model.labels_, return_counts=True))}
    print(f"Final model trained with k = {k}, cluster sizes {sizes}, saved to {model_path}")
    return {"k": k, "cluster_sizes": sizes, "model_path": model_path}


# ----------------- Task 8: Predict ----------------- #
def predict_clusters(model_path: str = MODEL_PATH, test_path: str = TEST_PATH) -> list:
    """
    Assign new customers to clusters using the saved model and scaler.

    Args:
        model_path (str): Path to the saved model bundle.
        test_path (str): CSV with the same FEATURES columns.
    Returns:
        list: One cluster id (int) per row in the test file.
    Raises:
        ValueError: If the test file is missing a required column.
    """
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    df = pd.read_csv(test_path)
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Test file missing columns: {missing}")

    X = bundle["scaler"].transform(np.log1p(df[bundle["features"]].values))
    preds = [int(p) for p in bundle["model"].predict(X)]
    print(f"Predicted clusters for {len(preds)} new customers: {preds}")
    return preds


# ----------------- Task 9: Report ----------------- #
def write_report(report: dict, run_id: str, report_dir: str = REPORT_DIR) -> str:
    """
    Save a JSON summary of the run so every DAG execution leaves an audit trail.

    Args:
        report (dict): Collected results from upstream tasks.
        run_id (str): Airflow run id, used in the file name.
        report_dir (str): Folder for report files.
    Returns:
        str: Path of the written report.
    """
    os.makedirs(report_dir, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() else "_" for ch in run_id)
    path = os.path.join(report_dir, f"report_{safe_id}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Run report written to {path}")
    return path
