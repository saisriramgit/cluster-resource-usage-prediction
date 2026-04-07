from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    node_count = 12
    time_steps = 700
    timestamps = pd.date_range("2025-01-01", periods=time_steps, freq="H")

    for node in range(node_count):
        base_cpu = rng.uniform(25, 55)
        base_mem = rng.uniform(30, 65)
        base_temp = rng.uniform(40, 55)

        for idx, ts in enumerate(timestamps):
            active_jobs = max(0, int(rng.normal(6 + node % 4, 2)))
            queue_depth = max(0, int(rng.normal(8 + active_jobs * 0.9, 3)))
            io_wait = max(0, rng.normal(4 + queue_depth * 0.25, 1.5))
            network_in = max(0, rng.normal(180 + active_jobs * 18, 35))
            network_out = max(0, rng.normal(165 + active_jobs * 16, 30))

            seasonal = 8 * np.sin(idx / 18)
            cpu_usage = np.clip(base_cpu + seasonal + active_jobs * 3.8 + io_wait * 1.2 + rng.normal(0, 6), 3, 100)
            memory_usage = np.clip(base_mem + active_jobs * 2.9 + queue_depth * 0.9 + rng.normal(0, 5), 5, 100)
            temperature = np.clip(base_temp + cpu_usage * 0.22 + rng.normal(0, 2), 25, 95)
            power_draw = np.clip(120 + cpu_usage * 2.7 + memory_usage * 0.8 + rng.normal(0, 12), 80, 450)

            overload_risk = 1 if (cpu_usage > 82 or memory_usage > 88 or temperature > 78 or queue_depth > 18) else 0

            if rng.random() < 0.02:
                cpu_usage = np.clip(cpu_usage + rng.uniform(18, 35), 0, 100)
                memory_usage = np.clip(memory_usage + rng.uniform(10, 24), 0, 100)
                temperature = np.clip(temperature + rng.uniform(6, 12), 0, 100)
                overload_risk = 1

            rows.append(
                {
                    "timestamp": ts,
                    "node_id": f"node_{node+1}",
                    "cpu_usage": round(float(cpu_usage), 2),
                    "memory_usage": round(float(memory_usage), 2),
                    "io_wait": round(float(io_wait), 2),
                    "network_in": round(float(network_in), 2),
                    "network_out": round(float(network_out), 2),
                    "active_jobs": active_jobs,
                    "queue_depth": queue_depth,
                    "temperature": round(float(temperature), 2),
                    "power_draw": round(float(power_draw), 2),
                    "failure_risk": overload_risk,
                }
            )

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cluster_telemetry.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote dataset to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
