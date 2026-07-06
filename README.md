# 🖥️ Enterprise Server Telemetry Predictor (ESTP)

An AIOps system that predicts enterprise server failure risks from continuous telemetry logs and schedules proactive maintenance — preventing costly unplanned outages before they happen.

> 🚀 **Live Demo:** [Open App on Streamlit Cloud](https://share.streamlit.io) *(Replace this with your deployed Streamlit App URL)*

---

## 📋 Problem & Constraints

This system simulates and validates an infrastructure management solution for **10 global clients** (each with 5–20 data centers and 5,000+ servers). It satisfies the following assignment constraints:

| Constraint | Target | Status |
|---|---|---|
| Downtime reduction | ≥ 40% | ✅ Demonstrated via backtest simulator |
| Client generalization | Cross-environment | ✅ Zero-shot + fine-tuning adaptation |
| Missing log tolerance | Up to 25% dropout | ✅ Median imputation + completeness feature |
| Prediction latency | < 5 seconds/server | ✅ < 0.1 ms/server (>100,000× margin) |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **UI Framework** | Streamlit |
| **ML Model** | Scikit-learn `GradientBoostingClassifier` |
| **Missing Data** | Scikit-learn `SimpleImputer` (median strategy) |
| **Visualizations** | Plotly (interactive charts) |
| **Language** | Python 3.12 |

### Features Used
- `memory_usage` — server RAM utilisation (%)
- `disk_io` — disk I/O operations per second
- `network_latency` — round-trip latency (ms)
- `error_rate` — Poisson-distributed error count
- `crash_count_7d` — crash frequency (last 7 days)
- `days_since_maintenance` — time since last scheduled audit
- `data_completeness` *(derived)* — `1.0 − fraction_of_missing_fields`

---

## 📊 Dashboard Tabs

### 1. Fleet Telemetry Predictor
Generate synthetic server telemetry (or upload a CSV) and score every server for failure probability. Results are classified into **Low / Medium / High** risk tiers, displayed in a sortable table, and available for CSV download.

### 2. Downtime Savings Simulator
Interactive backtest comparing **reactive** (run-to-failure) vs. **predictive** (ESTP-driven) maintenance strategies. Adjust fleet size, recovery times, and the decision threshold to explore the trade-off surface.

### 3. Generalization & Adaptation
Simulates deploying the global model on a **brand-new client** with drifted telemetry distributions. Shows zero-shot accuracy vs. accuracy after fine-tuning on a 40% calibration slice of the new client's data.

### 4. Robustness & Latency
Benchmarks model accuracy across missing-log rates (0–50%) and measures per-server inference latency across the fleet.

---

## ⚡ Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this folder as the root of a **public GitHub repository**.
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repository → set main file to `app.py`.
4. Click **Deploy** — the app will be live in minutes.

> **Note:** `packages.txt` (containing `libgomp1`) and `.python-version` (pinned to `3.12`) are included to ensure a stable build environment on Streamlit Cloud.

---

## 📁 Repository Structure

```
server-telemetry-predictor/
├── app.py               # Streamlit UI — 4-tab AIOps dashboard
├── model_utils.py       # Telemetry generation, feature engineering,
│                        #   GradientBoosting training & simulations
├── verify_project.py    # Automated constraint verification script
├── requirements.txt     # Python dependencies
├── packages.txt         # Linux system packages (libgomp1)
├── .python-version      # Python version pin (3.12)
└── .gitignore
```
