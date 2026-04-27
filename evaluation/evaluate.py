import time
import numpy as np
from utils.metrics import evaluate_predictions


def benchmark_detector(detector, test_df):
    latencies = []
    predictions = []

    for i in range(len(test_df)):
        row = test_df.iloc[[i]]

        start = time.perf_counter()
        pred = detector.predict(row)[0]
        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)
        predictions.append(pred)

    y_true = test_df["label"].values
    y_pred = np.array(predictions)

    metrics = evaluate_predictions(y_true, y_pred)

    metrics["avg_latency_ms"] = float(np.mean(latencies))
    metrics["p95_latency_ms"] = float(np.percentile(latencies, 95))
    metrics["max_latency_ms"] = float(np.max(latencies))

    return metrics