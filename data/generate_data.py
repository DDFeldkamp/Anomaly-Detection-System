import numpy as np
import pandas as pd
from config import SENSOR_COLUMNS, RANDOM_SEED, ANOMALY_RATE, N_SAMPLES


def generate_sensor_data(n_samples=N_SAMPLES, anomaly_rate=ANOMALY_RATE):
    np.random.seed(RANDOM_SEED)

    timestamps = np.arange(n_samples)

    motion = np.random.normal(0.25, 0.08, n_samples)
    audio = np.random.normal(35, 5, n_samples)
    vibration = np.random.normal(0.15, 0.05, n_samples)
    temperature = np.random.normal(22, 1.5, n_samples)
    door_open = np.random.binomial(1, 0.03, n_samples)

    labels = np.zeros(n_samples)

    anomaly_indices = np.random.choice(
        n_samples,
        size=int(n_samples * anomaly_rate),
        replace=False
    )

    for idx in anomaly_indices:
        anomaly_type = np.random.choice([
            "motion_spike",
            "audio_spike",
            "vibration_spike",
            "after_hours_entry",
            "sensor_dropout"
        ])

        labels[idx] = 1

        if anomaly_type == "motion_spike":
            motion[idx] += np.random.uniform(0.7, 1.2)

        elif anomaly_type == "audio_spike":
            audio[idx] += np.random.uniform(40, 70)

        elif anomaly_type == "vibration_spike":
            vibration[idx] += np.random.uniform(0.6, 1.0)

        elif anomaly_type == "after_hours_entry":
            door_open[idx] = 1
            motion[idx] += np.random.uniform(0.5, 1.0)
            audio[idx] += np.random.uniform(20, 40)

        elif anomaly_type == "sensor_dropout":
            motion[idx] = 0
            audio[idx] = 0
            vibration[idx] = 0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "motion": motion,
        "audio": audio,
        "vibration": vibration,
        "temperature": temperature,
        "door_open": door_open,
        "label": labels.astype(int)
    })

    return df


if __name__ == "__main__":
    df = generate_sensor_data()
    df.to_csv("sensor_data.csv", index=False)
    print(df.head())