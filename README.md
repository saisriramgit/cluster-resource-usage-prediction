# Cluster Resource Usage Prediction and Anomaly Detection

A machine learning project aligned with data-intensive computing, cloud/HPC systems, and performance-aware analytics.

This project focuses on system telemetry from compute nodes and applies machine learning to:

- predict future CPU utilization
- predict memory pressure
- detect anomalous resource behavior in distributed systems
- support data-intensive and scalable computing environments

## Features

- synthetic cluster telemetry generator (so the project runs immediately)
- supervised regression for CPU utilization prediction
- supervised classification for overload-risk detection
- unsupervised anomaly detection using Isolation Forest
- feature importance reporting
- plots for quick analysis

## Project structure

- `src/generate_data.py` - creates a realistic synthetic telemetry dataset
- `src/train.py` - trains models and saves reports/plots
- `data/` - input data folder
- `outputs/` - generated metrics, plots, and model files

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Run

```bash
python src/generate_data.py
python src/train.py
```

## Input columns

The generated dataset includes:

- timestamp
- node_id
- cpu_usage
- memory_usage
- io_wait
- network_in
- network_out
- active_jobs
- queue_depth
- temperature
- power_draw
- failure_risk

## Outputs

- `outputs/regression_metrics.json`
- `outputs/classification_metrics.json`
- `outputs/anomaly_summary.json`
- `outputs/cpu_prediction_scatter.png`
- `outputs/feature_importance.png`
- `outputs/anomaly_scores.png`
