"""
Automated Verification Script for Enterprise Server Telemetry Predictor (ESTP).
Asserts code execution, latency limits, missing data handling, and downtime reduction targets.
"""

import time
import numpy as np
from model_utils import train_model, predict_from_dataframe, simulate_downtime, generate_synthetic_telemetry

def run_tests():
    print("==================================================")
    print("RUNNING AUTOMATED VERIFICATION: ESTP CONSTRAINTS")
    print("==================================================")
    
    # 1. Train model and assert data loading
    print("[1/4] Training model with 25% missing data rate...")
    start_t = time.perf_counter()
    model, cols = train_model(missing_rate=0.25)
    train_dur = time.perf_counter() - start_t
    print(f"      Model trained successfully in {train_dur:.2f}s.")
    
    # 2. Verify Latency Constraint (< 5 seconds per server)
    print("[2/4] Testing prediction latency per server...")
    test_df = generate_synthetic_telemetry(n_clients=2, servers_per_client=500, missing_rate=0.25)
    
    start_t = time.perf_counter()
    preds = predict_from_dataframe(model, cols, test_df)
    eval_dur = time.perf_counter() - start_t
    
    avg_latency = eval_dur / len(test_df)
    print(f"      Scored {len(test_df)} servers in {eval_dur:.4f}s.")
    print(f"      Average scoring latency per server: {avg_latency * 1000:.4f} ms.")
    assert avg_latency < 5.0, f"Scoring latency {avg_latency:.4f}s exceeds 5.0s budget!"
    print("      [OK] PASSED: Prediction latency is comfortably < 5 seconds per server.")
    
    # 3. Verify Missing Data Tolerance (25% missing fields)
    print("[3/4] Testing missing data robustness...")
    null_frac = test_df[["memory_usage", "disk_io", "network_latency", "error_rate", "crash_count_7d"]].isna().mean().mean()
    print(f"      Telemetry data null fraction: {null_frac:.1%}")
    assert null_frac >= 0.20, f"Generated missing rate is too low: {null_frac:.1%}"
    
    assert "failure_probability" in preds.columns, "Predictions missing probability output!"
    assert not preds["failure_probability"].isna().any(), "Model returned NaN predictions!"
    print("      [OK] PASSED: Model successfully processed telemetry with 25% missing fields.")
    
    # 4. Verify Downtime Reduction Target (>= 40%)
    print("[4/4] Evaluating simulated downtime reduction...")
    # Use economically optimal threshold of 0.15 on a larger fleet to stabilize estimate
    sim = simulate_downtime(model, cols, n_servers=4000, proactive_downtime_h=1.0, reactive_downtime_h=8.0, threshold=0.15, missing_rate=0.25)
    reduction = sim["reduction_pct"]
    print(f"      Simulated downtime reduction: {reduction:.1%} (Prevented: {sim['prevented_failures']}/{sim['total_failures']} failures)")
    assert reduction >= 0.40, f"Downtime reduction {reduction:.1%} is below the 40% threshold!"
    print("      [OK] PASSED: Model-driven scheduling achieves >= 40% downtime reduction.")
    
    print("\n==================================================")
    print("ALL ESTP CONSTRAINTS VERIFIED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
