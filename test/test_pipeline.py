import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.pipeline import (
    FEATURES,
    compute_elbow,
    compute_silhouette,
    decode,
    encode,
    predict_clusters,
    preprocess_data,
    quality_gate,
    select_k,
    train_final_model,
    validate_data,
    write_report,
)


# ----------------- Fixtures ----------------- #
@pytest.fixture
def raw_df():
    """1200 synthetic customers in 4 clear groups, positive values like real amounts."""
    X, _ = make_blobs(n_samples=1200, centers=4, n_features=len(FEATURES), random_state=42)
    X = np.abs(X) * 500
    return pd.DataFrame(X, columns=FEATURES)


@pytest.fixture
def prep_b64(raw_df):
    return preprocess_data(encode(raw_df))


# ----------------- Validation ----------------- #
def test_validate_data_passes(raw_df):
    summary = validate_data(encode(raw_df))
    assert summary["rows"] == 1200
    assert set(summary["null_ratio"]) == set(FEATURES)


def test_validate_data_missing_column(raw_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(encode(raw_df.drop(columns=["PAYMENTS"])))


def test_validate_data_too_few_rows(raw_df):
    with pytest.raises(ValueError, match="rows"):
        validate_data(encode(raw_df.head(50)))


def test_validate_data_too_many_nulls(raw_df):
    raw_df.loc[:200, "BALANCE"] = np.nan
    with pytest.raises(ValueError, match="Null ratio"):
        validate_data(encode(raw_df))


# ----------------- Preprocessing ----------------- #
def test_preprocess_standardizes(prep_b64):
    X, scaler = decode(prep_b64)
    assert X.shape == (1200, len(FEATURES))
    assert np.allclose(X.mean(axis=0), 0, atol=1e-6)
    assert hasattr(scaler, "transform")


# ----------------- Model selection ----------------- #
def test_elbow_and_silhouette_return_valid_k(prep_b64):
    elbow = compute_elbow(prep_b64)
    sil = compute_silhouette(prep_b64, sample_size=500)
    assert 2 <= elbow["k_elbow"] <= 15
    assert 2 <= sil["k_silhouette"] <= 15
    assert all(-1 <= s <= 1 for s in sil["scores"].values())


def test_select_k_keeps_elbow_within_tolerance():
    sel = select_k({"k_elbow": 4}, {"k_silhouette": 5, "scores": {"4": 0.40, "5": 0.42}})
    assert sel["chosen_k"] == 4


def test_select_k_overrides_elbow_outside_tolerance():
    sel = select_k({"k_elbow": 4}, {"k_silhouette": 5, "scores": {"4": 0.20, "5": 0.42}})
    assert sel["chosen_k"] == 5


def test_select_k_missing_score_raises():
    with pytest.raises(ValueError):
        select_k({"k_elbow": 9}, {"k_silhouette": 5, "scores": {"5": 0.42}})


# ----------------- Quality gate ----------------- #
def test_quality_gate_pass():
    assert quality_gate({"silhouette": 0.30}, threshold=0.25) == "train_final_model_task"


def test_quality_gate_fail():
    assert quality_gate({"silhouette": 0.10}, threshold=0.25) == "quality_alert_task"


# ----------------- Train, predict, report ----------------- #
def test_train_and_predict_round_trip(prep_b64, raw_df, tmp_path):
    model_path = str(tmp_path / "bundle.pkl")
    test_path = str(tmp_path / "test.csv")
    raw_df.head(3).to_csv(test_path, index=False)

    result = train_final_model(prep_b64, {"chosen_k": 4}, model_path)
    preds = predict_clusters(model_path, test_path)

    assert result["k"] == 4
    assert sum(result["cluster_sizes"].values()) == 1200
    assert len(preds) == 3 and all(0 <= p < 4 for p in preds)


def test_write_report(tmp_path):
    path = write_report({"status": "ok"}, "manual__2026-10-09T15:00:00+00:00", str(tmp_path))
    with open(path) as f:
        assert json.load(f)["status"] == "ok"
