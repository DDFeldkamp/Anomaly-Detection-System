from data.generate_data import generate_sensor_data
from models.isolation_forest import IsolationForestDetector
from models.autoencoder import AutoencoderDetector
from evaluation.evaluate import benchmark_detector
from streaming.realtime_detector import RealTimeSensorStream, RealTimeDetector


def train_test_split(df, train_ratio=0.7):
    split = int(len(df) * train_ratio)

    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    train_normal = train_df[train_df["label"] == 0]

    return train_normal, test_df

def print_metrics_table(metrics, title="Model Metrics"):
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50)

    print(f"{'Metric':<25} | {'Value':>15}")
    print("-" * 50)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<25} | {value:>15.4f}")
        else:
            print(f"{key:<25} | {value:>15}")

    print("=" * 50)


def main():
    df = generate_sensor_data()

    train_df, test_df = train_test_split(df)

    print("Training Isolation Forest...")
    isolation_detector = IsolationForestDetector(contamination=0.08)
    isolation_detector.fit(train_df)

    print("Benchmarking Isolation Forest...")
    iso_metrics = benchmark_detector(isolation_detector, test_df)
    print("Isolation Forest Metrics:")
    print_metrics_table(iso_metrics, "Isolation Forest Metrics")

    print("\nTraining Autoencoder...")
    autoencoder_detector = AutoencoderDetector(threshold_percentile=95)
    autoencoder_detector.fit(train_df)

    print("Benchmarking Autoencoder...")
    ae_metrics = benchmark_detector(autoencoder_detector, test_df)
    print("Autoencoder Metrics:")
    print_metrics_table(ae_metrics, "Autoencoder Metrics")

    print("\nRunning real-time stream demo...")
    stream = RealTimeSensorStream(test_df.head(200), delay_seconds=0.001)
    real_time_detector = RealTimeDetector(autoencoder_detector)
    alerts = real_time_detector.run(stream)

    print(f"\nTotal alerts generated: {len(alerts)}")


if __name__ == "__main__":
    main()