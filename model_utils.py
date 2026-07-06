"""
Core ML logic for the Enterprise Server Telemetry Predictor (ESTP).

Approach: GradientBoosting classifier (scikit-learn) trained on server telemetry
(memory, disk I/O, network latency, error rate, crash count, days since
maintenance). Handles missing values via median imputation (up to 25%+ missing logs)
and responds in milliseconds. Includes a timeline backtest simulator
for proactive maintenance to demonstrate a >40% reduction in downtime.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RNG = np.random.default_rng(42)

FEATURE_COLS = [
    "memory_usage", "disk_io", "network_latency",
    "error_rate", "crash_count_7d", "days_since_maintenance",
]


def generate_synthetic_telemetry(n_clients=10, servers_per_client=400, missing_rate=0.25, anomaly_shift=0.0):
    """Simulates multi-client server telemetry with a realistic failure signal.
    Allows injecting an anomaly_shift (mean shift in telemetry features) to represent
    a new/different infrastructure setup for generalization testing.
    """
    rows = []
    for client_id in range(n_clients):
        n_dcs = RNG.integers(5, 21)
        for _ in range(servers_per_client):
            dc_id = RNG.integers(0, n_dcs)

            # Base distributions shifted by anomaly_shift for client diversity
            memory_usage = RNG.normal(60 + anomaly_shift * 5, 15)
            disk_io = RNG.normal(50 + anomaly_shift * 8, 20)
            network_latency = RNG.normal(30 + anomaly_shift * 10, 10)
            error_rate = RNG.poisson(2 + max(0.0, anomaly_shift * 1.5))
            crash_count_7d = RNG.poisson(0.5 + max(0.0, anomaly_shift * 0.5))
            days_since_maintenance = RNG.integers(0, 180)

            # Bounds validation
            memory_usage = np.clip(memory_usage, 0, 100)
            disk_io = np.clip(disk_io, 0, 150)
            network_latency = np.clip(network_latency, 0, 1000)

            risk = (
                0.20 * memory_usage + 0.15 * disk_io + 0.10 * network_latency
                + 3.0 * error_rate + 5.0 * crash_count_7d
                + 0.15 * days_since_maintenance + RNG.normal(0, 0.4)
            )
            rows.append(dict(
                client_id=client_id, data_center_id=f"{client_id}-{dc_id}",
                memory_usage=memory_usage, disk_io=disk_io,
                network_latency=network_latency, error_rate=error_rate,
                crash_count_7d=crash_count_7d,
                days_since_maintenance=days_since_maintenance, risk=risk,
            ))
    df = pd.DataFrame(rows)
    threshold = df["risk"].quantile(0.85)
    prob_fail = np.where(df["risk"] > threshold, 0.65, 0.04)
    df["failed_next_48h"] = (RNG.random(len(df)) < prob_fail).astype(int)
    df = df.drop(columns=["risk"])

    # Simulate missing logs
    if missing_rate > 0:
        for col in ["memory_usage", "disk_io", "network_latency", "error_rate", "crash_count_7d"]:
            mask = RNG.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan
    return df


def add_completeness_feature(df):
    """Adds a data-completeness score so the model can gauge confidence
    when telemetry is partially missing."""
    df = df.copy()
    telemetry_cols = ["memory_usage", "disk_io", "network_latency", "error_rate", "crash_count_7d"]
    df["data_completeness"] = 1.0 - df[telemetry_cols].isna().mean(axis=1)
    return df


def train_model(df=None, missing_rate=0.25):
    """Trains a GradientBoosting failure-prediction pipeline on telemetry data.
    Uses a median imputer so missing values are handled gracefully.
    """
    if df is None:
        df = generate_synthetic_telemetry(missing_rate=missing_rate)
    df = add_completeness_feature(df)
    cols = FEATURE_COLS + ["data_completeness"]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4,
            random_state=42,
        )),
    ])
    pipeline.fit(df[cols], df["failed_next_48h"])
    return pipeline, cols


def predict_from_dataframe(model, cols, input_df):
    """Runs predictions on a dataframe of server telemetry.
    Fills missing schema columns as NaN to allow evaluation.
    """
    input_df = input_df.copy()
    for c in FEATURE_COLS:
        if c not in input_df.columns:
            input_df[c] = np.nan
    input_df = add_completeness_feature(input_df)

    probs = model.predict_proba(input_df[cols])[:, 1]
    result = input_df.copy()
    result["failure_probability"] = probs
    result["risk_level"] = pd.cut(
        probs, bins=[-0.01, 0.15, 0.5, 1.01], labels=["Low", "Medium", "High"]
    )
    return result


def simulate_downtime(model, cols, n_servers=2000, proactive_downtime_h=1.0, reactive_downtime_h=8.0, threshold=0.45, missing_rate=0.25):
    """Simulates a maintenance backtest timeline to show reactive vs predictive downtime.
    Returns structured metrics demonstrating downtime savings.
    """
    # Generate test fleet
    df = generate_synthetic_telemetry(n_clients=2, servers_per_client=n_servers // 2, missing_rate=missing_rate)
    pred_df = predict_from_dataframe(model, cols, df)

    # Ground truth actual failures
    actual_failures = pred_df["failed_next_48h"].values
    predicted_probs = pred_df["failure_probability"].values

    # Baseline: Reactive Maintenance (Only fix after failure)
    total_failures = int(actual_failures.sum())
    reactive_downtime = total_failures * reactive_downtime_h

    # Predictive Strategy
    flagged_mask = predicted_probs >= threshold

    # Count outcomes
    true_positives = int(((flagged_mask == 1) & (actual_failures == 1)).sum())
    false_positives = int(((flagged_mask == 1) & (actual_failures == 0)).sum())
    false_negatives = int(((flagged_mask == 0) & (actual_failures == 1)).sum())

    # Predictive downtime calculation
    # TP + FP cost proactive maintenance downtime. FN cost reactive downtime.
    proactive_actions = true_positives + false_positives
    predictive_downtime = (proactive_actions * proactive_downtime_h) + (false_negatives * reactive_downtime_h)

    downtime_saved = reactive_downtime - predictive_downtime
    reduction_pct = downtime_saved / max(1.0, reactive_downtime)

    return {
        "total_servers": len(df),
        "total_failures": total_failures,
        "prevented_failures": true_positives,
        "unprevented_failures": false_negatives,
        "false_alarms": false_positives,
        "reactive_downtime_h": reactive_downtime,
        "predictive_downtime_h": predictive_downtime,
        "downtime_saved_h": downtime_saved,
        "reduction_pct": reduction_pct,
    }


def adapt_and_evaluate_new_client(model, cols, test_servers=500, anomaly_shift=1.5, missing_rate=0.25):
    """Simulates a 'Cold Start' new client scenario with shifted hardware parameters.
    Evaluates:
    - Zero-Shot: Base model running directly.
    - Adapted: Fine-tuned model using a small calibration dataset from the new client.
    """
    # 1. Generate new client data
    new_client_df = generate_synthetic_telemetry(n_clients=1, servers_per_client=test_servers, missing_rate=missing_rate, anomaly_shift=anomaly_shift)

    # Split into calibration (40%) and evaluation (60%)
    split_idx = int(len(new_client_df) * 0.4)
    calib_df = new_client_df.iloc[:split_idx].copy()
    eval_df = new_client_df.iloc[split_idx:].copy()

    # Add completeness
    calib_df = add_completeness_feature(calib_df)
    eval_df = add_completeness_feature(eval_df)

    # 2. Zero-Shot Predictions on Eval set
    zero_shot_res = predict_from_dataframe(model, cols, eval_df)
    zero_shot_preds = (zero_shot_res["failure_probability"] >= 0.15).astype(int)
    zero_shot_acc = (zero_shot_preds == eval_df["failed_next_48h"]).mean()

    # 3. Adaptation (incremental training or refitting on a blend of old + new calib data)
    # Generate historical data for background reference
    hist_df = generate_synthetic_telemetry(n_clients=5, servers_per_client=200, missing_rate=missing_rate)
    hist_df = add_completeness_feature(hist_df)

    # Blend old and calibration data (weighting the new environment)
    blended_df = pd.concat([hist_df, calib_df, calib_df], ignore_index=True)

    adapted_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4,
            random_state=42,
        )),
    ])
    adapted_pipeline.fit(blended_df[cols], blended_df["failed_next_48h"])

    # Predict on Eval set with adapted model
    adapted_res = predict_from_dataframe(adapted_pipeline, cols, eval_df)
    adapted_preds = (adapted_res["failure_probability"] >= 0.15).astype(int)
    adapted_acc = (adapted_preds == eval_df["failed_next_48h"]).mean()

    return {
        "zero_shot_accuracy": float(zero_shot_acc),
        "adapted_accuracy": float(adapted_acc),
        "new_client_failures": int(eval_df["failed_next_48h"].sum()),
        "eval_size": len(eval_df)
    }
