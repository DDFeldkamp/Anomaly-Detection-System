import time


class RealTimeSensorStream:
    def __init__(self, df, delay_seconds=0.001):
        self.df = df
        self.delay_seconds = delay_seconds

    def stream(self):
        for _, row in self.df.iterrows():
            time.sleep(self.delay_seconds)
            yield row


class RealTimeDetector:
    def __init__(self, detector):
        self.detector = detector
        self.alerts = []

    def run(self, stream):
        for row in stream.stream():
            row_df = row.to_frame().T

            prediction = self.detector.predict(row_df)[0]
            score = self.detector.anomaly_score(row_df)[0]

            if prediction == 1:
                alert = {
                    "timestamp": int(row["timestamp"]),
                    "score": float(score),
                    "reason": "anomalous sensor pattern"
                }

                self.alerts.append(alert)
                print(f"[ALERT] timestamp={alert['timestamp']} score={alert['score']:.4f}")

        return self.alerts